use parking_lot::*;
use smallvec::alloc::collections::VecDeque;
use smallvec::alloc::sync::Arc;
use std::sync::atomic::*;
use std::thread::JoinHandle;
use std::time::Duration;

pub struct Task {
    pub closure: Box<dyn FnOnce() -> () + Send>,
    pub high_priority: bool,
    /// Use only for very high-priority tasks
    pub allowed_on_main_thread: bool,
}

impl Task {
    pub fn new<F: FnOnce() -> () + Send + 'static>(
        closure: F,
        high_priority: bool,
        allowed_on_main_thread: bool,
    ) -> Self {
        Self {
            closure: Box::new(closure),
            high_priority,
            allowed_on_main_thread,
        }
    }

    pub fn execute(self) {
        (self.closure)()
    }

    /// Creates a compound task that runs itself, and then the other task
    pub fn compose_with(self, other: Self) -> Self {
        let high_priority = self.high_priority | other.high_priority;
        let allowed_on_main_thread = self.allowed_on_main_thread | other.allowed_on_main_thread;
        Self::new(
            move || {
                self.execute();
                other.execute();
            },
            high_priority,
            allowed_on_main_thread,
        )
    }
}

struct TaskPoolRunnerParams {
    queue: Arc<(Mutex<VecDeque<Task>>, Condvar)>,
    kill_switch: Arc<AtomicBool>,
}

fn task_pool_runner(params: TaskPoolRunnerParams) {
    while !params.kill_switch.load(Ordering::Acquire) {
        let mut queue = params.queue.0.lock();
        if let Some(task) = queue.pop_front() {
            drop(queue);
            task.execute();
        } else {
            params.queue.1.wait(&mut queue);
        }
    }
}

pub struct TaskPool {
    threads: Vec<JoinHandle<()>>,
    queue: Arc<(Mutex<VecDeque<Task>>, Condvar)>,
    kill_switch: Arc<AtomicBool>,
}

impl TaskPool {
    pub fn new(thread_count: usize) -> Self {
        let mut pool = Self {
            threads: Vec::with_capacity(thread_count),
            queue: Arc::new(Default::default()),
            kill_switch: Arc::new(Default::default()),
        };
        pool.change_thread_count(thread_count);
        pool
    }

    fn kill_threads(&mut self) {
        if !self.threads.is_empty() {
            self.kill_switch.store(true, Ordering::SeqCst);
            self.queue.1.notify_all();
            for thread in self.threads.drain(..) {
                thread.join().expect("Couldn't join worker thread");
            }
            self.kill_switch.store(false, Ordering::SeqCst);
        }
    }

    pub fn change_thread_count(&mut self, new_thread_count: usize) {
        let new_thread_count = new_thread_count.max(1).min(num_cpus::get() * 2);
        if self.threads.len() != new_thread_count {
            self.kill_threads();
            for wid in 0..new_thread_count {
                let params = TaskPoolRunnerParams {
                    queue: self.queue.clone(),
                    kill_switch: self.kill_switch.clone(),
                };
                let jh = std::thread::Builder::new()
                    .name(format!("ballxworld-worker-{}", wid))
                    .stack_size(4 * 1024 * 1024) // 4 MiB stack for each worker
                    .spawn(move || {
                        task_pool_runner(params);
                    })
                    .expect("Could not spawn worker thread");
                self.threads.push(jh);
            }
        }
    }

    pub fn main_thread_tick(&self) {
        if let Some(mut queue) = self.queue.0.try_lock_for(Duration::from_micros(500)) {
            if let Some(task) = queue.front() {
                if task.allowed_on_main_thread {
                    let task = queue.pop_front().unwrap();
                    drop(queue);
                    task.execute();
                }
            }
        }
    }

    pub fn push_tasks<TaskIter: Iterator<Item = Task>>(&self, tasks: TaskIter) {
        let mut tasks_added = 0usize;
        let mut queue = self.queue.0.lock();
        for task in tasks {
            if task.high_priority {
                queue.push_front(task);
            } else {
                queue.push_back(task);
            }
            tasks_added += 1;
        }
        drop(queue);
        if tasks_added * 3 < self.threads.len() {
            for _ in 0..tasks_added {
                self.queue.1.notify_one();
            }
        } else {
            self.queue.1.notify_all();
        }
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        self.kill_threads();
    }
}

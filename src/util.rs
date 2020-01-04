#[macro_export]
macro_rules! offset_of {
    ($ty:ty, $field:ident $(,)?) => {
        unsafe {
            let null: *const $ty = std::ptr::null();
            std::mem::transmute::<_, usize>(&(*null).$field as *const _)
        }
    };
}

fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/world.capnp")
        .run().expect("schema compiler command");
}

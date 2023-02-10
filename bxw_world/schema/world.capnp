
@0xca2d18da1692381c;

struct NameRegistryEntry {
    id @0 :UInt64;
    namespace @1 :Text = "bxw";
    name @2 :Text;
}

struct NameRegistry {
    activeMappings @0 :List(NameRegistryEntry);
    removedMappings @1 :List(NameRegistryEntry);
}

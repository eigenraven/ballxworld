use cgmath::prelude::*;
use cgmath::{vec3, Quaternion, Vector3};
use std::collections::HashMap;
use std::marker::PhantomData;

#[repr(u64)]
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum EntityDomain {
    LocalChunked = 0b00_u64 << 62,
    SharedChunked = 0b01_u64 << 62,
    LocalOmnipresent = 0b10_u64 << 62,
    SharedOmnipresent = 0b11_u64 << 62,
}

impl EntityDomain {
    pub fn number(self) -> usize {
        (self as u64 >> 62) as usize
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ValidEntityID(u64);

pub type EntityID = Option<ValidEntityID>;

const DOMAIN_MASK: u64 = 0b11_u64 << 62;

impl ValidEntityID {
    pub fn from_raw(r: u64) -> EntityID {
        if r & !DOMAIN_MASK == 0 {
            None
        } else {
            Some(ValidEntityID(r))
        }
    }

    pub fn from_parts(domain: EntityDomain, sub_id: u64) -> EntityID {
        assert_eq!(sub_id & DOMAIN_MASK, 0);
        if sub_id == 0 {
            None
        } else {
            Some(ValidEntityID(domain as u64 | sub_id))
        }
    }

    pub fn u64(self) -> u64 {
        self.0
    }

    pub fn domain(self) -> EntityDomain {
        unsafe { std::mem::transmute(self.u64() & DOMAIN_MASK) }
    }

    pub fn sub_id(self) -> u64 {
        self.u64() & !DOMAIN_MASK
    }
}

pub trait Component {
    fn name() -> &'static str;
    fn entity_id(&self) -> ValidEntityID;
}

#[derive(Clone, Debug)]
pub enum BoundingShape {
    Point,
    Ball { r: f32 },
    Box { size: Vector3<f32> },
}

#[derive(Clone, Debug)]
pub struct CLocation {
    id: ValidEntityID,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub orientation: Quaternion<f32>,
    pub bounding_shape: BoundingShape,
    pub bounding_offset: Vector3<f32>,
}

impl CLocation {
    pub fn new(id: ValidEntityID) -> Self {
        Self {
            id,
            position: vec3(0.0, 0.0, 0.0),
            velocity: vec3(0.0, 0.0, 0.0),
            orientation: Quaternion::one(),
            bounding_shape: BoundingShape::Point,
            bounding_offset: vec3(0.0, 0.0, 0.0),
        }
    }
}

impl Component for CLocation {
    fn name() -> &'static str {
        "Location"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug)]
pub struct CDebugInfo {
    id: ValidEntityID,
    pub ent_name: String,
}

impl CDebugInfo {
    pub fn new(id: ValidEntityID, ent_name: String) -> Self {
        Self { id, ent_name }
    }
}

impl Component for CDebugInfo {
    fn name() -> &'static str {
        "DebugInfo"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug)]
pub struct CLoadAnchor {
    id: ValidEntityID,
    pub radius: u32,
}

impl CLoadAnchor {
    pub fn new(id: ValidEntityID, radius: u32) -> Self {
        Self { id, radius }
    }
}

impl Component for CLoadAnchor {
    fn name() -> &'static str {
        "LoadAnchor"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ComponentId<T>(usize, PhantomData<T>);

impl<T> ComponentId<T> {
    pub fn new(i: usize) -> Self {
        Self(i, PhantomData)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: ValidEntityID,
    pub location: Option<ComponentId<CLocation>>,
    pub debug_info: Option<ComponentId<CDebugInfo>>,
    pub load_anchor: Option<ComponentId<CLoadAnchor>>,
}

impl Entity {
    fn new(id: ValidEntityID) -> Self {
        Self {
            id,
            location: None,
            debug_info: None,
            load_anchor: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ECS {
    // ECS variables
    entities: HashMap<ValidEntityID, Entity>,
    last_nonfree_ids: [u64; 4],
    // components
    locations: Vec<CLocation>,
    debug_infos: Vec<CDebugInfo>,
    load_anchors: Vec<CLoadAnchor>,
}

impl ECS {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn iter(&self) -> std::collections::hash_map::Values<'_, ValidEntityID, Entity> {
        self.entities.values()
    }

    pub fn iter_mut(&mut self) -> std::collections::hash_map::ValuesMut<'_, ValidEntityID, Entity> {
        self.entities.values_mut()
    }

    pub fn add_new_entity(&mut self, domain: EntityDomain) -> ValidEntityID {
        let nfi = domain.number();
        let mut sub_id = self.last_nonfree_ids[nfi] + 1;
        while self
            .entities
            .contains_key(&ValidEntityID::from_parts(domain, sub_id).unwrap())
        {
            sub_id += 1;
        }
        self.last_nonfree_ids[nfi] = sub_id;
        let id = ValidEntityID::from_parts(domain, sub_id).unwrap();
        let iret = self.entities.insert(id, Entity::new(id));
        assert!(iret.is_none());
        id
    }

    #[allow(clippy::map_entry)]
    pub fn add_entity_with_id(&mut self, raw_id: u64) -> Result<ValidEntityID, ()> {
        let id = ValidEntityID::from_raw(raw_id).ok_or(())?;
        let sub_id = id.sub_id();
        let nfi = id.domain().number();
        if self.last_nonfree_ids[nfi] < sub_id {
            self.last_nonfree_ids[nfi] = sub_id;
        }
        if self.entities.contains_key(&id) {
            Err(())
        } else {
            self.entities.insert(id, Entity::new(id));
            Ok(id)
        }
    }
}

pub trait ECSHandler<C: Component> {
    fn has_component(&self, e: ValidEntityID) -> bool;
    fn get_component(&self, e: ValidEntityID) -> Option<&C>;
    fn get_component_mut(&mut self, e: ValidEntityID) -> Option<&mut C>;
    fn set_component(&mut self, e: ValidEntityID, c: C);
    fn iter(&self) -> std::slice::Iter<C>;
    fn iter_mut(&mut self) -> std::slice::IterMut<C>;
}

macro_rules! impl_ecs_fns {
    ( $t:ty, $snake_name:ident, $plural_name:ident ) => {
        impl ECSHandler<$t> for ECS {
            fn has_component(&self, e: ValidEntityID) -> bool {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0]).is_some()
            }

            fn get_component(&self, e: ValidEntityID) -> Option<&$t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0])
            }

            fn get_component_mut(&mut self, e: ValidEntityID) -> Option<&mut $t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(move |i| &mut self.$plural_name[i.0])
            }

            fn iter(&self) -> std::slice::Iter<$t> {
                self.$plural_name.iter()
            }

            fn iter_mut(&mut self) -> std::slice::IterMut<$t> {
                self.$plural_name.iter_mut()
            }

            fn set_component(&mut self, e: ValidEntityID, c: $t) {
                let cid = self
                    .entities
                    .get(&e)
                    .clone()
                    .and_then(|e| e.$snake_name.clone());
                match cid {
                    Some(cid) => {
                        self.$plural_name[cid.0] = c;
                    }
                    None => {
                        let cid = ComponentId::new(self.$plural_name.len());
                        self.$plural_name.push(c);
                        self.entities.get_mut(&e).unwrap().$snake_name = Some(cid);
                    }
                }
            }
        }
    };
}

impl_ecs_fns!(CLocation, location, locations);
impl_ecs_fns!(CDebugInfo, debug_info, debug_infos);
impl_ecs_fns!(CLoadAnchor, load_anchor, load_anchors);

use crate::ChunkPosition;
use bxw_util::zstd;
use std::cell::RefCell;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SerializedChunkData {
    pub position: ChunkPosition,
    pub voxel_data: Vec<u8>,
    pub entity_data: Vec<u8>,
}

thread_local! {
static ZSTD_COMPRESSOR: RefCell<zstd::bulk::Compressor<'static>> = RefCell::new(zstd::bulk::Compressor::new(STORAGE_ZSTD_COMPRESS_LEVEL).unwrap());
static ZSTD_DECOMPRESSOR: RefCell<zstd::bulk::Decompressor<'static>> = RefCell::new(zstd::bulk::Decompressor::new().unwrap());
}
const STORAGE_ZSTD_COMPRESS_LEVEL: i32 = 8;
const STORAGE_ZSTD_DECOMPRESS_SIZE_LIMIT: usize = 4 * 1024 * 1024;

pub fn storage_zstd_compress(data: &[u8]) -> Vec<u8> {
    ZSTD_COMPRESSOR
        .with(|c| c.borrow_mut().compress(data))
        .expect("Unexpected compression error")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecompressError {
    UnknownError,
}

pub fn storage_zstd_decompress(
    data: &[u8],
    custom_size_limit: Option<usize>,
) -> Result<Vec<u8>, DecompressError> {
    ZSTD_DECOMPRESSOR
        .with(|c| {
            c.borrow_mut().decompress(
                data,
                custom_size_limit.unwrap_or(STORAGE_ZSTD_DECOMPRESS_SIZE_LIMIT),
            )
        })
        .map_err(|_| DecompressError::UnknownError)
}

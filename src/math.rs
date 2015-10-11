//! Assorted mathematical helper functions.

use conv::{
    ValueInto
};

/// Helper for a conversion that we know can't fail.
pub fn cast<T, U>(x: T) -> U where T: ValueInto<U> {
    match x.value_into() {
        Ok(y) => y,
        Err(_) => panic!("Failed to convert"),
    }
}

use cargo_metadata::MetadataCommand;
use semver::Version;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(lagacy_image)");
    println!("cargo:rerun-if-changed=Cargo.lock");

    let metadata = MetadataCommand::new()
        .exec()
        .expect("Failed to fetch metadata");

    let image_version = &metadata
        .packages
        .iter()
        .find(|p| p.name == "image")
        .expect("image crate not found in dependencies")
        .version;

    if image_version < &Version::parse("0.25.8").unwrap() {
        println!("cargo:rustc-cfg=lagacy_image");
    }
}

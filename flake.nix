{
  description = "A very basic flake";

  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      # nixpkgs-unstable:
      rev = "1b5c1881789eb8c86c655caeff3c918fb76fbfe6";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = false;
          };
        };
        rustNightly = pkgs.rust-bin.nightly."2025-10-21".default.override {
          extensions = [
            "clippy"
            "miri"
            "rustfmt"
            "rust-src"
          ];
        };
      in {
        devShells = {
          default = pkgs.mkShell {
            packages =
              [
                rustNightly
              ]
              ++ (with pkgs; [
                git
              ]);
          };
        };
      }
    );
}

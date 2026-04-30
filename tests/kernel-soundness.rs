use imageproc::kernel::Kernel;

#[test]
#[should_panic]
fn kernel_new_rejects_overflowing_dimensions() {
    let data = [0.0f32; 2];
    let _ = Kernel::new(&data, 2, 2_147_483_649);
}

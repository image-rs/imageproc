use imageproc::filter::Kernel;

#[test]
#[should_panic]
fn kernel_new_rejects_overflowing_dimensions() {
    // 2 * 2_147_483_649 overflows u32 to 2, matching data.len().
    // The widened u64 check must catch this.
    let data = [0.0f32; 2];
    let _ = Kernel::new(&data, 2, 2_147_483_649);
}

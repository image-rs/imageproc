extern crate sdl2; 
extern crate image;

use image::RgbaImage;
use sdl2::event::WindowEvent;
use sdl2::rect::Rect;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::surface::Surface;

fn display_image(title: &str, image: &RgbaImage, window_width: i32, window_height: i32) {

    fn get_scale(win_width: i32, win_height: i32, image_width: u32, image_height: u32) -> f32{
        let width_scale = win_width as f32 / image_width as f32; 
        let height_scale = win_height as f32 / image_height as f32;
        if  width_scale < height_scale {
            width_scale
        } else {
            height_scale
        }
    };

    fn get_output_image(scale: f32, img: &RgbaImage) -> (u32, u32, RgbaImage) {
        let height = (scale * img.height() as f32) as u32;
        let width = (scale * img.width() as f32) as u32;
        let output_image = image::imageops::resize(img, width, height, image::FilterType::Triangle); 
        (width, height, output_image)
    };

    let image_width = image.width();
    let image_height = image.height();

    let (width, height, output_image) = 
        if image_height < window_height as u32 && image_width < window_width as u32 {
            (image_width, image_height, image.clone())
        } else {
            let scale = get_scale(window_width, window_height, image_width, image_height);
            get_output_image(scale, image)
        };

    const CHANNEL_COUNT: u32 = 4;
    let pitch  = width * CHANNEL_COUNT;
    let mut img_raw = output_image.into_raw();
    let surface_img = Surface::from_data(
        &mut img_raw, 
        width, 
        height,
        pitch, 
        PixelFormatEnum::ABGR8888  // this format is necessary because sdl2 expects bits from highest to lowest
    ).expect("couldn't converted image to surface");
    let sdl = sdl2::init().expect("couldn't create sdl2 context");
    let video_subsystem = sdl.video().expect("couldn't create video subsystem");
    let mut window = video_subsystem
        .window(title, window_width as u32, window_height as u32)
        .position_centered()
        .resizable()
        .build()
        .expect("window couldn't be created");
    window.set_minimum_size(300, 300).expect("invalid minimum size for window");

    let mut canvas = window.into_canvas().build().expect("Couldn't create CanvasBuilder");
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator.create_texture_from_surface(surface_img)
        .expect("couldn't create texture from surface");

    let center_x = ((window_width as u32 - width) as f32 / 2.0_f32) as i32;
    let center_y = ((window_height as u32 - height) as f32 / 2.0_f32) as i32; 
    canvas.set_draw_color(Color::RGB(255, 255, 255));
    canvas.clear();
    canvas.copy(&texture, None, Rect::new(center_x, center_y, width, height)).unwrap();
    canvas.present();
    let mut event_pump = sdl.event_pump().unwrap();
    event_pump.enable_event(sdl2::event::EventType::Window);
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                    Event::KeyDown {keycode: Some(Keycode::Escape), .. } => {
                        break 'running
                    },
                Event::Window { win_event: WindowEvent::Resized(x, y), .. } => {
                    
                    // resize image if necessary to fit into the window
                    let (width, height, output_image) = 
                        if image_height < y as u32 && image_width < x as u32 {
                            (image_width, image_height, image.clone())
                        } else {
                            let scale = get_scale(x, y, image_width, image_height);
                            get_output_image(scale, image)
                        };

                    let pitch  = width * CHANNEL_COUNT;
                    let mut img_raw = output_image.into_raw();
                    let surface_img = Surface::from_data(
                        &mut img_raw, 
                        width, 
                        height,
                        pitch, 
                        PixelFormatEnum::ABGR8888  // this format is necessary because sdl2 expects bits from highest to lowest
                    ).expect("couldn't converted image to surface");

                    texture = texture_creator.create_texture_from_surface(surface_img)
                        .expect("couldn't create texture from surface");
                
                    println!("x: {}, y: {}\nwidth: {}, height: {}", x, y, width, height);
                    let center_x = ((x as u32 - width) as f32 / 2.0_f32) as i32;
                    let center_y = ((y as u32 - height) as f32 / 2.0_f32) as i32; 
                    canvas.clear(); 
                    canvas.copy(&texture, None, Rect::new(center_x, center_y, width, height)).unwrap();
                    canvas.present();
                },           
                _ => {}
            }
        }
    }
    // #[cfg(test)]
    // mod tests {
        // use super::display_image;
        // use super::display_image;

        #[test]
        fn test_get_scale() {
        
            let img_path = {
                std::env::args().nth(1).expect("no argument provided for image path")
            };

            let img = image::open("images/colors.jpg")
                .expect("couldn't read image from that file path").to_rgba();
            let width = 1000;
            let height = 900;
            // let temp_img = image::imageops::resize(&img, 500, 500, image::FilterType::Triangle); 
            display_image("ma image", &img, width, height);
            assert_eq!(get_scale(5, 5, 5, 5), 1.0);
        }    

    // }
}


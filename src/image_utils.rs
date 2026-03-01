// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Image utility functions for processing and converting images

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use image::ImageFormat;
use std::io::Cursor;

/// Convert a PNG base64 string to a JPG base64 string.
///
/// Returns None if the input is invalid or conversion fails.
pub fn convert_png_b64_to_jpg_b64(png_b64_str: Option<&str>) -> Option<String> {
    let png_b64 = png_b64_str?;

    if png_b64.len() < 10 {
        eprintln!(
            "⚠️  Invalid base64 string (too short): {}",
            &png_b64[..std::cmp::min(50, png_b64.len())]
        );
        return None;
    }

    let img_bytes = match BASE64.decode(png_b64) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("❌ Error decoding base64: {}", e);
            return None;
        }
    };

    let img = match image::load_from_memory(&img_bytes) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("❌ Error loading image: {}", e);
            return None;
        }
    };

    let mut jpg_buf = Cursor::new(Vec::new());
    match img.write_to(&mut jpg_buf, ImageFormat::Jpeg) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("❌ Error converting to JPEG: {}", e);
            return None;
        }
    }

    Some(BASE64.encode(jpg_buf.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb, Rgba, RgbaImage};
    use std::io::Cursor;

    fn make_png_b64(width: u32, height: u32, r: u8, g: u8, b: u8) -> String {
        let img = ImageBuffer::from_fn(width, height, |_, _| Rgb([r, g, b]));
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        BASE64.encode(buf.into_inner())
    }

    #[test]
    fn test_valid_png_converts_to_jpg() {
        let png_b64 = make_png_b64(10, 10, 255, 0, 0);
        let result = convert_png_b64_to_jpg_b64(Some(&png_b64));
        assert!(result.is_some());
        // Verify it's valid JPEG
        let jpg_bytes = BASE64.decode(result.unwrap()).unwrap();
        let img = image::load_from_memory(&jpg_bytes).unwrap();
        assert_eq!(img.color(), image::ColorType::Rgb8);
    }

    #[test]
    fn test_rgba_png_converts_correctly() {
        let img: RgbaImage = ImageBuffer::from_fn(10, 10, |_, _| Rgba([255, 0, 0, 128]));
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        let png_b64 = BASE64.encode(buf.into_inner());

        let result = convert_png_b64_to_jpg_b64(Some(&png_b64));
        assert!(result.is_some());
        // JPEG doesn't support alpha, verify it's RGB
        let jpg_bytes = BASE64.decode(result.unwrap()).unwrap();
        let img = image::load_from_memory(&jpg_bytes).unwrap();
        assert_eq!(img.color(), image::ColorType::Rgb8);
    }

    #[test]
    fn test_none_input_returns_none() {
        let result = convert_png_b64_to_jpg_b64(None);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_string_returns_none() {
        let result = convert_png_b64_to_jpg_b64(Some(""));
        assert!(result.is_none());
    }

    #[test]
    fn test_short_string_returns_none() {
        let result = convert_png_b64_to_jpg_b64(Some("abc"));
        assert!(result.is_none());
    }

    #[test]
    fn test_invalid_base64_returns_none() {
        let result =
            convert_png_b64_to_jpg_b64(Some("not_valid_base64_image_data_that_is_long_enough"));
        assert!(result.is_none());
    }

    #[test]
    fn test_preserves_image_dimensions() {
        let png_b64 = make_png_b64(50, 30, 0, 255, 0);
        let result = convert_png_b64_to_jpg_b64(Some(&png_b64));
        assert!(result.is_some());
        let jpg_bytes = BASE64.decode(result.unwrap()).unwrap();
        let img = image::load_from_memory(&jpg_bytes).unwrap();
        assert_eq!(img.width(), 50);
        assert_eq!(img.height(), 30);
    }
}

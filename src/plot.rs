use anyhow::Result;
use plotters::{
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};
use std::collections::HashMap;

pub fn plot_bigram_heatmap<T: Into<f64> + Copy>(
    b: &HashMap<(String, String), T>,
    chars: &[String],
    char_to_idx: &HashMap<String, usize>,
    output_path: &str,
    title: &str,
) -> Result<()> {
    let n = chars.len();

    // Create the heatmap data
    let mut data = vec![vec![0.0; n]; n];
    for ((ch1, ch2), count) in b {
        let i = char_to_idx[ch1];
        let j = char_to_idx[ch2];
        data[i][j] = (*count).into();
    }

    let root = BitMapBackend::new(output_path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = data.iter().flatten().fold(0.0_f64, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(60)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(-0.5f32..(n as f32 - 0.5), (n as f32 - 0.5)..(-0.5f32))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(n)
        .y_labels(n)
        .x_label_style(("sans-serif", 15))
        .y_label_style(("sans-serif", 15))
        .x_label_formatter(&|x| chars[x.round() as usize].clone())
        .y_label_formatter(&|y| chars[y.round() as usize].clone())
        .draw()?;

    let plotting_area = chart.plotting_area();
    for i in 0..n {
        for j in 0..n {
            let value = data[i][j];
            if value > 0.0 {
                let color = RGBColor(
                    255,
                    ((1.0 - value / max_val) * 255.0) as u8,
                    ((1.0 - value / max_val) * 255.0) as u8,
                );
                plotting_area.draw(&Rectangle::new(
                    [
                        (j as f32 - 0.5, i as f32 - 0.5),
                        (j as f32 + 0.5, i as f32 + 0.5),
                    ],
                    color.filled(),
                ))?;

                plotting_area.draw(&Text::new(
                    format!("{}{}", chars[i], chars[j]),
                    (j as f32, i as f32 - 0.2),
                    ("sans-serif", 10)
                        .into_font()
                        .color(&BLACK)
                        .pos(Pos::new(HPos::Center, VPos::Center)),
                ))?;

                plotting_area.draw(&Text::new(
                    if value >= 1.0 {
                        format!("{}", value as i32)
                    } else {
                        format!("{:.3}", value)
                    },
                    (j as f32, i as f32 + 0.2),
                    ("sans-serif", 10)
                        .into_font()
                        .color(&BLACK)
                        .pos(Pos::new(HPos::Center, VPos::Center)),
                ))?;
            }
        }
    }

    root.present()?;
    println!("Heatmap saved as {}", output_path);
    Ok(())
}

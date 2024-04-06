pub mod map_loader;

pub mod map_gen {
    use pathfinding::prelude::*;
    pub fn random_grid(width: usize, height: usize, seed: u64) -> Grid {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut grid = Grid::new(width, height);
        for x in 0..width {
            for y in 0..height {
                if rng.gen_bool(0.2) {
                    grid.add_vertex((x, y));
                }
            }
        }
        grid.invert();
        grid.enable_diagonal_mode();
        grid
    }
}

pub mod plot_tools {
    pub mod colours {
        use plotters::prelude::HSLColor;
        pub fn lerp_hsv(a: HSLColor, b: HSLColor, t: f64) -> HSLColor {
            let t = t.min(1.).max(0.);
            let (mut h_a, h_b) = match a.0 > b.0 {
                true => (b.0, a.0),
                false => (a.0, b.0),
            };

            let hue = match h_b - h_a {
                d if d > 0.5 => {
                    h_a += 1.;
                    (h_a + t * (h_b - h_a)) % 1.
                }
                d if d <= 0.5 => h_a + t * d,
                _ => 0.,
            };
            HSLColor(hue, a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2))
        }
    }
    pub mod plots {
        use super::colours;
        use ndarray::Array2;
        use pathfinding::grid::Grid;
        use plotters::prelude::*;
        pub fn plot_grid(
            grid: &Grid,
            drawing_area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
            start: &(usize, usize),
            goal: &(usize, usize),
            path: Option<&Vec<(usize, usize)>>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let sub_areas = drawing_area.split_evenly((grid.height, grid.width));
            const BLUE: plotters::style::RGBColor = RGBColor(59, 132, 227);
            let mut idx: usize = 0;
            (0..grid.height).for_each(|y| {
                (0..grid.width).for_each(|x| {
                    if grid.has_vertex((x, y)) {
                        let _ = sub_areas[idx].fill(&WHITE);
                        let _ = sub_areas[idx].titled(&format!("{idx}"), ("sans-serif", 12));
                    } else {
                        let _ = sub_areas[idx].fill(&full_palette::GREY_A200);
                        let _ = sub_areas[idx].titled(&format!("{idx}"), ("sans-serif", 12));
                    }
                    idx += 1;
                })
            });

            if let Some(path) = path {
                for &(x, y) in path {
                    sub_areas[y * grid.width + x].fill(&BLUE)?;
                }
            }

            sub_areas[start.1 * grid.width + start.0].fill(&GREEN)?;
            sub_areas[goal.1 * grid.width + goal.0].fill(&RED)?;

            Ok(())
        }

        pub fn prob_as_heatmap(
            probs: &Array2<f64>,
            drawing_area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            drawing_area.fill(&BLUE)?;
            let sub_areas = drawing_area.split_evenly((probs.nrows(), probs.ncols()));
            const LOW: plotters::style::HSLColor =
                plotters::style::HSLColor(172. / 360., 0.66, 0.85);
            const HIGH: plotters::style::HSLColor =
                plotters::style::HSLColor(351. / 360., 0.67, 0.45);
            let mut idx: usize = 0;
            for y in 0..probs.nrows() {
                for x in 0..probs.ncols() {
                    let prob = probs.row(y)[x];
                    match sub_areas[idx].fill(&colours::lerp_hsv(LOW, HIGH, prob)) {
                        Ok(()) => {}
                        _ => {
                            println!("Failed to fill with colour")
                        }
                    }
                    sub_areas[idx].titled(&format!("{prob:.2}"), ("sans-serif", 12))?;
                    idx += 1;
                }
            }
            Ok(())
        }
    }
    pub mod save_output {
        use super::plots;
        use ndarray::Array2;
        use pathfinding::grid::Grid;
        use plotters::prelude::*;

        pub fn save_heatmap(
            probs: &Array2<f64>,
            file_name: &str,
            title: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            // let _ = plot_path(result.unwrap().0);
            const OUT_FILE_DIR: &str = "plotters-doc-data/";
            let out_file_name: String = [OUT_FILE_DIR, file_name].join("");
            let root = BitMapBackend::new(&out_file_name, (1920, 1080)).into_drawing_area();
            root.fill(&RGBColor(100, 255, 255))?;
            let root = root
                .titled(title, ("sans-serif", 60))?
                .shrink(((1920 - 1080) / 2, 0), (1000, 1000));
            plots::prob_as_heatmap(probs, &root)?;

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", out_file_name);
            Ok(())
        }

        pub fn save_result(
            result: Option<Vec<(usize, usize)>>,
            grid: Grid,
            start: (usize, usize),
            goal: (usize, usize),
            file_name: &str,
            title: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            // let _ = plot_path(result.unwrap().0);
            const OUT_FILE_DIR: &str = "plotters-doc-data/";
            let out_file_name: String = [OUT_FILE_DIR, file_name].join("");
            let root = BitMapBackend::new(&out_file_name, (1920, 1080)).into_drawing_area();
            root.fill(&RGBColor(100, 100, 100))?;
            let root = root
                .titled(title, ("sans-serif", 60))?
                .shrink(((1920 - 1080) / 2, 0), (1000, 1000));
            match result {
                Some(path) => plots::plot_grid(&grid, &root, &start, &goal, Some(&path))?,
                None => plots::plot_grid(&grid, &root, &start, &goal, None)?,
            }

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", out_file_name);
            Ok(())
        }
    }
}

pub mod search_algorithms;

pub mod probabilistic;

pub mod detection {
    use pathfinding::grid::Grid;

    pub fn detect_absolute(
        ground: &Grid,
        perceived: &mut Grid,
        &(x, y): &(usize, usize),
        kernel: ndarray::Array2<bool>,
    ) {
        // The Kernel should be centred on the location
        // x - nrows//2 + i
        // y - ncols//2 + j
        assert!(kernel.ncols() % 2 == 1, "Kernel has no middle column");
        assert!(kernel.nrows() % 2 == 1, "Kernel has no middle row");
        for i in 0..kernel.nrows() {
            for j in 0..kernel.ncols() {
                if kernel[[i, j]] {
                    let row = x - kernel.nrows() / 2 + i;
                    let col = y - kernel.ncols() / 2 + j;
                    if (0..perceived.width).contains(&row) && (0..perceived.height).contains(&col) {
                        match ground.has_vertex((row, col)) {
                            true => perceived.add_vertex((row, col)),
                            false => perceived.remove_vertex((row, col)),
                        };
                    }
                }
            }
        }
    }
}

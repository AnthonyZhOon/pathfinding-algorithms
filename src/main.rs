use plotters::style::full_palette::GREY_A200;

use ndarray::prelude::*;

use hello_grid::map_gen::random_grid;
use hello_grid::map_loader::read_map;
use pathfinding::prelude::astar;
use pathfinding::prelude::Grid;
use plotters::coord::Shift;
use plotters::prelude::*;

fn plot_grid(
    grid: &Grid,
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
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
                let _ = sub_areas[idx].fill(&GREY_A200);
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

fn detect_absolute(
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

fn detect_uncertain(
    _ground: &Grid,
    _perceived_probs: &mut Array2<f64>,
    &(x, y): &(usize, usize),
    kernel: ndarray::Array2<f64>,
) {
    assert!(kernel.ncols() % 2 == 1, "Kernel has no middle column");
    assert!(kernel.nrows() % 2 == 1, "Kernel has no middle row");
    for i in 0..kernel.nrows() {
        for j in 0..kernel.ncols() {
            if x + j >= kernel.nrows() / 2 {
                y + i >= kernel.ncols();
            }
            let _row = x - kernel.nrows() / 2 + i;
            let _col = y - kernel.ncols() / 2 + j;
            // if (0..perceived_probs.nrows()).contains(&row) && (0..perceived_probs.ncols()).contains(&col)   {
            //     match ground.has_vertex((row, col)) {
            //         true => perceived_probs.add_vertex((row, col)),
            //         false => perceived_probs.remove_vertex((row, col)),
            //     };}
        }
    }
}

fn convolve_2d(matrix: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let mut res = Array::from_elem((matrix.nrows(), matrix.ncols()), 0.);
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    // let mut idx = 0;
    // save_heatmap(matrix, "Convolve_Ground.png", "Truth").ok();
    for m_col in 0..matrix.ncols() {
        for m_row in 0..matrix.nrows() {
            for k_col in 0..kernel.ncols() {
                for k_row in 0..kernel.nrows() {
                    if m_row + k_row >= kernel.nrows() / 2 && m_col + k_col >= kernel.ncols() / 2 {
                        let row = m_row + k_row - kernel.nrows() / 2;
                        let col = m_col + k_col - kernel.ncols() / 2;
                        if col < cols && row < rows {
                            res[[m_row, m_col]] += kernel[[k_row, k_col]] * matrix[[row, col]];
                        }
                    }
                    // save_heatmap(&res, &format!("Convolve_{idx}.png"), "Lies").ok();
                    // idx += 1;
                }
            }
        }
    }
    res
}

fn blur_to_prob(grid: &Grid) -> Array2<f64> {
    // Consider multiple iterations of blurring
    let arr = {
        let mut tmp = vec![0.; grid.width * grid.height];
        grid.iter().for_each(|(x, y)| tmp[x + y * grid.width] = 1.);
        tmp
    };
    let base = Array::from_shape_vec((grid.height, grid.width), arr).unwrap();
    convolve_2d(&base, &Array::from_elem((3, 3), 1. / 9.))
}

fn sample_prob_grid(probs: &Array2<f64>) -> Grid {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut grid = Grid::new(probs.ncols(), probs.nrows());
    for y in 0..probs.nrows() {
        for x in 0..probs.ncols() {
            if rng.gen_bool(probs[[y, x]].min(1.)) {
                grid.add_vertex((x, y));
            }
        }
    }
    grid
}

fn lerp_hsv(a: HSLColor, b: HSLColor, t: f64) -> HSLColor {
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

fn prob_as_heatmap(
    probs: &Array2<f64>,
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
) -> Result<(), Box<dyn std::error::Error>> {
    drawing_area.fill(&BLUE)?;
    let sub_areas = drawing_area.split_evenly((probs.nrows(), probs.ncols()));
    const LOW: plotters::style::HSLColor = plotters::style::HSLColor(172. / 360., 0.66, 0.85);
    const HIGH: plotters::style::HSLColor = plotters::style::HSLColor(351. / 360., 0.67, 0.45);
    let mut idx: usize = 0;
    for y in 0..probs.nrows() {
        for x in 0..probs.ncols() {
            let prob = probs.row(y)[x];
            match sub_areas[idx].fill(&lerp_hsv(LOW, HIGH, prob)) {
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rand::Rng;
    // let grid = random_grid(100, 100, 139);
    let mut grid = read_map("maps/warehouse_small.map");
    grid.disable_diagonal_mode();
    let mut rng = rand::thread_rng();
    let goal = grid
        .iter()
        .nth(rng.gen_range(0..grid.vertices_len()))
        .unwrap();
    let start = grid
        .iter()
        .nth(rng.gen_range(0..grid.vertices_len()))
        .unwrap();
    let successors_weighted = |p: &(usize, usize)| {
        grid.neighbours(*p)
            .iter()
            .map(|&p| (p, 1))
            .collect::<Vec<_>>()
    };
    let success = |p: &(usize, usize)| *p == goal;
    let heuristic = |p: &(usize, usize)| grid.distance(*p, goal);
    let result_astar: Option<(Vec<(usize, usize)>, usize)> =
        astar(&start, successors_weighted, heuristic, |p| *p == goal);
    match result_astar {
        Some((path, _cost)) => save_result(
            Some(path),
            grid.clone(),
            start,
            goal,
            "astar.png",
            "Ground truth",
        )?,
        None => save_result(
            None,
            grid.clone(),
            start,
            goal,
            "astar.png",
            "Ground Truth (Impossible)",
        )?,
    }

    // Take the ground truth and run a convolution with noise to create a probability interpretation of the grid

    // Use the probability map to instantiate world-hypotheses and plan on them

    // Step through the proposed solutions, voting on the next move and updating agent's map of the world?
    // Replan either every step or when encountering newly perceived terrain/lack of
    //
    // Run the simulation
    let mut current = start;
    let mut island = Grid::new(3, 3);
    island.add_vertex((0, 0));
    island.add_vertex((2, 2));
    let island_prob = blur_to_prob(&island);
    // panic!();
    let probs = blur_to_prob(&grid);

    save_heatmap(&island_prob, "island_prob.png", "Probabilities as Heatmap")?;
    // save_result(None, island, (0,0), (2,2), "island.png", "1");
    let sampled_guess = sample_prob_grid(&probs);
    // for i in 0..10 {
    //     save_result(None, sample_prob_grid(&prob), (0,0), (0,0), &format!("island_sample{i}.png"), &format!("Island {i}"));
    // }
    let mut guess = sampled_guess.clone();
    let _ = save_result(
        None,
        sampled_guess.clone(),
        (0, 0),
        (0, 0),
        "sampled.png",
        "sample",
    );
    save_heatmap(&probs, "Probabilities.png", "Probabilities as Heatmap")?;
    guess.add_vertex(goal);
    guess.add_vertex(start);
    let mut steps: usize = 0;
    while !success(&current) {
        // TODO:
        // Detect can update the underlying probability map (measure and predict cycles)
        // Visualise nicer, side-by-side with ground truth, show "fog of war", show heatmap of probabilities
        // Visualise taken trajectory
        // Create maps from files, taken from warthog?
        // Detect can see through walls currently??

        let guess_astar = astar(
            &current,
            |&n| {
                guess
                    .neighbours(n)
                    .iter()
                    .map(|&p| (p, 1))
                    .collect::<Vec<_>>()
            },
            |&_n| guess.distance(current, goal),
            |p| *p == goal,
        );

        match guess_astar {
            Some((path, _cost)) => {
                save_result(
                    Some(path.clone()),
                    guess.clone(),
                    start,
                    goal,
                    &format!("astar_{steps}.png"),
                    &format!("Guessed path {steps}"),
                )?;
                current = *path.iter().take(2).last().unwrap()
            }
            None => {
                save_result(
                    None,
                    guess.clone(),
                    start,
                    goal,
                    &format!("astar_{steps}.png"),
                    &format!("Guessed path {steps} (impossible)"),
                )?;
                break;
            }
        }
        // Take 1 step and update position and visible graph
        detect_absolute(
            &grid,
            &mut guess,
            &current,
            ndarray::Array::from_elem((11, 11), true),
        );

        steps += 1;
    }
    println!("Done");

    // save_result(None, map_loader::read_map("maps/warehouse_small.map"), (0, 0), (0, 0), "warehouse.png", "warehouse small");
    // let result_bfs: Option<Vec<(usize, usize)>> = bfs(start, &mut successors, &mut success);

    // match result_bfs {
    //     Some(path) => save_result(Some(path), grid.clone(), start, goal, "bfs.png")?,
    //     None => save_result(None, grid.clone(), start, goal, "bfs.png")?
    // }

    Ok(())
}

fn save_heatmap(
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
    prob_as_heatmap(probs, &root)?;

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file_name);
    Ok(())
}

fn save_result(
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
        Some(path) => plot_grid(&grid, &root, &start, &goal, Some(&path))?,
        None => plot_grid(&grid, &root, &start, &goal, None)?,
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file_name);
    Ok(())
}

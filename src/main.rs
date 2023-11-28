use pathfinding::num_traits::PrimInt;
use pathfinding::prelude::astar;
use pathfinding::prelude::Grid;
use plotters::coord::Shift;
use plotters::prelude::*;

fn plot_grid(
    grid: &Grid,
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
    start: &(usize, usize),
    goal: &(usize, usize),
    path: Option<&Vec<(usize, usize)>>
)-> Result<(), Box<dyn std::error::Error>>{
    let sub_areas = drawing_area.split_evenly((grid.width, grid.height));
    const BLUE: plotters::style::RGBColor = RGBColor(59, 132, 227);
    for (idx, sub_area) in (0..).zip(sub_areas.iter()) {
        if grid.has_vertex((idx % grid.width, idx / grid.width)) {
            sub_area.fill(&WHITE)?;
            // sub_area.titled(&idx.to_string(), ("sans-serif", 30))?;
        } else {
            sub_area.fill(&GREY_A200)?;
        }
    }
    if let Some(path) = path {
        for &(x, y) in path {
            sub_areas[x*grid.width + y].fill(&BLUE)?;
        }
    }
    
    sub_areas[start.0*grid.width+start.1].fill(&GREEN)?;
    sub_areas[goal.0*grid.width+goal.1].fill(&RED)?;
    // println!("{:?}", path);

    Ok(())
}

fn random_grid(width: usize, height: usize, seed: u64) -> Grid {
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

pub mod search_algorithms;
use plotters::style::full_palette::GREY_A200;
use search_algorithms::bfs::bfs;
use search_algorithms::dfs::dfs;
use ndarray;
use ndarray::Array;
use ndarray::prelude::*;

fn detect(ground: &Grid, perceived: &mut Grid, &(x, y): &(usize, usize), kernel: ndarray::Array2<bool>) {
    // The Kernel should be centred on the location
    // x - nrows//2 + i
    // y - ncols//2 + j
    assert!(kernel.ncols() % 2 == 1, "Kernel has no middle column");
    assert!(kernel.nrows() % 2 == 1, "Kernel has no middle row");
    for i in 0..kernel.nrows() {
        for j in 0..kernel.ncols() {
            if kernel[[i, j]] {
                let col = x - kernel.nrows()/2 + i;
                let row = y - kernel.ncols()/2 + j;
                if (0..perceived.width).contains(&row) && (0..perceived.height).contains(&col)   {
                    match ground.has_vertex((row, col)) {
                        true => perceived.add_vertex((row, col)),
                        false => perceived.remove_vertex((row, col)),
                    };}
            }
        }
    }
}



fn blur_to_prob(grid: &Grid) -> Array2<f64> {
    // Consider multiple iterations of blurring
    use convolve2d::*;
    let arr = {
        let mut tmp = vec![0.; grid.width*grid.height];
        grid.iter().for_each(|(x, y)| tmp[x + y*grid.width] = 1.);
        tmp
    };
    let base = DynamicMatrix::new(grid.width, grid.height, arr).unwrap();
    let x: DynamicMatrix<f64> = convolve2d(&base, &DynamicMatrix::new(3, 3,
                                                 vec![1./9.; 9]).unwrap());
    ndarray::Array::from_shape_vec((grid.width, grid.height), x.get_data().into()).unwrap()
}

fn sample_prob_grid(probs: &Array2<f64>) -> Grid {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut grid = Grid::new(probs.nrows(), probs.ncols());
    for row in 0..probs.nrows(){
        for col in 0..probs.ncols() {

            if rng.gen_bool(probs.row(row)[col].min(1.)) {
                grid.add_vertex((row, col));
            }
        }
    }
    grid
}
fn main() -> Result<(), Box<dyn std::error::Error>> {

    use rand::Rng;
    let grid = random_grid(100, 100, 139);
    let mut rng = rand::thread_rng();
    let goal = grid.iter().nth(rng.gen_range(0..grid.vertices_len())).unwrap();
    let start = grid.iter().nth(rng.gen_range(0..grid.vertices_len())).unwrap();
    let successors_weighted = |p: &(usize, usize)| grid.neighbours(*p).iter().map(|&p| (p, 1)).collect::<Vec<_>>();
    let mut successors = |p: &(usize, usize)| grid.neighbours(*p);
    let mut success = |p: &(usize, usize)| *p == goal;
    let mut heuristic = |p: &(usize, usize)| grid.distance(*p, goal);
    let result_astar:Option<(Vec<(usize, usize)>, usize)>  = astar(
                        &start,
                         successors_weighted,
                         heuristic,
                          |p| *p == goal
                        );
    match result_astar {
        Some((path, _cost)) => save_result(Some(path), grid.clone(), start, goal, "astar.png", "Ground truth")?,
        None => save_result(None, grid.clone(), start, goal, "astar.png", "Ground Truth (Impossible)")?
    }

    // Take the ground truth and run a convolution with noise to create a probability interpretation of the grid

    // Use the probability map to instantiate world-hypotheses and plan on them

    // Step through the proposed solutions, voting on the next move and updating agent's map of the world?
    // Replan either every step or when encountering newly perceived terrain/lack of
    // 
    // Run the simulation
    let mut current = start;
    let mut guess = random_grid(100, 100, 2);
    let mut island = Grid::new(3,3);
    island.add_vertex((1,1));
    let mut prob = blur_to_prob(&grid);

    // save_result(None, island, (0,0), (2,2), "island.png", "1");
    let mut sampled_guess = sample_prob_grid(&prob); 
    // for i in 0..10 {
    //     save_result(None, sample_prob_grid(&prob), (0,0), (0,0), &format!("island_sample{i}.png"), &format!("Island {i}"));
    // }
    let mut guess = sampled_guess.clone();
    let _ = save_result(None, sampled_guess.clone(), (0,0), (0, 0), "sampled.png", "sample");

    guess.add_vertex(goal);
    guess.add_vertex(start);
    let mut steps: usize = 0;
    while !success(&current) {
        let guess_astar = astar(&current, |&n| guess.neighbours(n).iter().map(|&p| (p, 1)).collect::<Vec<_>>(), |&n| guess.distance(current, goal), |p| *p == goal);

        match guess_astar {
            Some((path, _cost)) => {save_result(Some(path.clone()), guess.clone(), start, goal, &format!("astar_{steps}.png"), &format!("Guessed path {steps}"))?;
                                                                current = *path.iter().take(2).last().unwrap()},
            None => {save_result(None, guess.clone(), start, goal, &format!("astar_{steps}.png"), &format!("Guessed path {steps} (impossible)"))?; break},
        }
        // Take 1 step and update position and visible graph
        detect(&grid, &mut guess, &current, ndarray::Array::from_elem((11,11), true));

        steps += 1;
    }
    println!("Done");
    
                
    // let result_bfs: Option<Vec<(usize, usize)>> = bfs(start, &mut successors, &mut success);

    // match result_bfs {
    //     Some(path) => save_result(Some(path), grid.clone(), start, goal, "bfs.png")?,
    //     None => save_result(None, grid.clone(), start, goal, "bfs.png")?
    // }

    Ok(())
}

fn save_result(
    result: Option<Vec<(usize, usize)>>,
    grid: Grid, start: (usize, usize),
    goal: (usize, usize),
    file_name: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // let _ = plot_path(result.unwrap().0);
    const OUT_FILE_DIR: &str = "plotters-doc-data/";
    let out_file_name: String = [OUT_FILE_DIR, file_name].join("");
    let root = BitMapBackend::new(&out_file_name, (1920, 1080)).into_drawing_area();
    root.fill(&RGBColor(100,100,100))?;
    let root = root
    .titled(title, ("sans-serif", 60))?
    .shrink(((1920 - 1080) / 2, 0), (1000, 1000));
    match result {
        Some(path) => plot_grid(&grid, &root, &start, &goal, Some(&path))?,
        None => plot_grid(&grid, &root, &start, &goal, None)?
    }

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file_name);
    Ok(())
}

mod safe_interval_astar {
    use pathfinding::prelude::Grid;
    use super::random_grid;
    use std::collections::HashMap;
    type Pos = (usize, usize);
    // Generate a grid, apply the dynamic obstacles to create safe intervals
    // Run djikstra_all for a heuristic
    // Create the neighbours function
    struct PosWithIntervals {
        position: Pos,
        safe_intervals: Vec<(usize, usize)>
    }

    struct NodeWithInterval(Pos, usize);

    fn prepare_grid() {
        let grid: Grid = random_grid(300, 300, 10);
        let mut interval_grid: HashMap<Pos, PosWithIntervals> = HashMap::new();
        for vertex in grid.iter() {
            interval_grid.insert(vertex, PosWithIntervals { position: vertex, safe_intervals: vec![(0, 99)] });
        }
    }

    

}


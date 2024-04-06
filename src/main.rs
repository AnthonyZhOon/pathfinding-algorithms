use hello_grid::detection::detect_absolute;
use hello_grid::plot_tools::save_output;
use hello_grid::probabilistic;

use hello_grid::map_loader;

use pathfinding::prelude::astar;
use pathfinding::prelude::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rand::Rng;
    // let grid = random_grid(100, 100, 139);
    let mut grid = map_loader::read_map("maps/warehouse_small.map");
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
        Some((path, _cost)) => save_output::save_result(
            Some(path),
            grid.clone(),
            start,
            goal,
            "astar.png",
            "Ground truth",
        )?,
        None => save_output::save_result(
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
    let island_prob = probabilistic::blur_to_prob(&island);
    // panic!();
    let probs = probabilistic::blur_to_prob(&grid);

    save_output::save_heatmap(&island_prob, "island_prob.png", "Probabilities as Heatmap")?;
    // save_result(None, island, (0,0), (2,2), "island.png", "1");
    let sampled_guess = probabilistic::sample_prob_grid(&probs);
    // for i in 0..10 {
    //     save_result(None, sample_prob_grid(&prob), (0,0), (0,0), &format!("island_sample{i}.png"), &format!("Island {i}"));
    // }
    let mut guess = sampled_guess.clone();
    let _ = save_output::save_result(
        None,
        sampled_guess.clone(),
        (0, 0),
        (0, 0),
        "sampled.png",
        "sample",
    );
    save_output::save_heatmap(&probs, "Probabilities.png", "Probabilities as Heatmap")?;
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
                save_output::save_result(
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
                save_output::save_result(
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

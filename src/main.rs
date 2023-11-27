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
    let red = 150;
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
            if rng.gen_bool(0.76) {
                grid.add_vertex((x, y));
            }
        }
    }
    grid.enable_diagonal_mode();
    grid
}

pub mod search_algorithms;
use plotters::style::full_palette::GREY_A200;
use search_algorithms::bfs::bfs;
use search_algorithms::dfs::dfs;
use ndarray;

fn detect(ground: Grid, mut perceived: Grid, (x, y): (usize, usize), kernel: ndarray::Array2<bool>) {
    // The Kernel should be centred on the location
    // x - nrows//2 + i
    // y - ncols//2 + j
    for i in 0..kernel.nrows() {
        for j in 0..kernel.ncols() {
            if kernel[[i, j]] {
                let row = x - kernel.nrows()/2 + i;
                let col = y - kernel.ncols()/2 + j;
                if row in  {
                    match ground.has_vertex((row, col)) {
                        true => ,
                        false =>,
                    }}
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    use rand::Rng;
    let grid = random_grid(100, 100, 131);
    let mut rng = rand::thread_rng();
    let goal = grid.iter().nth(rng.gen_range(0..grid.vertices_len())).unwrap();
    let start = grid.iter().nth(rng.gen_range(0..grid.vertices_len())).unwrap();
    let successors_weighted = |p: &(usize, usize)| grid.neighbours(*p).iter().map(|&p| (p, 1)).collect::<Vec<_>>();
    let mut successors = |p: &(usize, usize)| grid.neighbours(*p);
    let mut success = |p: &(usize, usize)| *p == goal;
    let mut heuristic = |p| grid.distance(*p, goal);
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
    let mut current = start.clone();
    let guess = random_grid(100, 100, 2);
    let mut steps: usize = 0;
    while !success(&current) {
        let guess_astar = astar(&start, successors_weighted, heuristic, |p| *p == goal);

        match guess_astar {
            Some((path, _cost)) => {save_result(Some(path), guess.clone(), start, goal, "astar.png", &format!("Guessed path {steps}"))?;
                                                                current = path.first().unwrap().clone()},
            None => {save_result(None, guess.clone(), start, goal, "astar.png", &format!("Guessed path {steps} (impossible)"))?; break},
        }
        // Take 1 step and update position and visible graph
        
        steps += 1;
    }

    
                
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

mod safe_interval_path_planning {
    use pathfinding::prelude::Grid;
    use std::collections::HashMap;
    type Node = (usize, usize);
    type Time = usize;
    type NodeWithInterval = (Node, usize, Time);
    #[derive(Clone)]
    struct Cfg(Node, usize, Time);
    type Interval = (Time, Time);
    type Heuristic = usize;
    /// Time is discretised to and uses f64
    /// Locations are nodes and is kept as a grid and uses (usize, usize)
    /// The OPEN set contains nodes with intervals, a unique state is a tuple describing location, interval i and arrival time t
    /// with heuristic + distance estimate as the score
    /// The Expanded set contains the fully relaxed nodes with g(s) finalised for each state of (location, interval, time)
    /// 
    enum DynamicEnvironment {
        Static(Grid),
        SafeIntervals((Grid, HashMap<Node,(Heuristic, Vec<Interval>)>)),
    }
    type TimedPath = Vec<NodeWithInterval>;

    fn sipp<FN, FH, FS, IN>(start: &Node,mut successors: FN,mut heuristic: FH,mut success: FS) -> Option<(TimedPath, Time)>
    where 
        FN: FnMut(&Node) -> IN,
        FH: FnMut(&Node) -> Time,
        FS: FnMut(&Node) -> bool,
        IN: IntoIterator<Item = NodeWithInterval>,
    {
        let result = sipp_core(start,&mut successors,&mut heuristic, &mut success);
        None
    }
    use slotmap::SlotMap;
    use slotmap::new_key_type;
    use std::collections::BinaryHeap;
    
    new_key_type! {struct NodeKey;}
    fn sipp_core<FN, FH, FS, IN>(start: &Node, successors: &mut FN, heuristic: &mut FH,success: &mut FS) -> Option<(Vec<Cfg>, Time)>
    where
        FN: FnMut(&Node) -> IN,
        FH: FnMut(&Node) -> Time,
        FS: FnMut(&Node) -> bool,
        IN: IntoIterator<Item = NodeWithInterval>,
    {
        
        let mut to_see = BinaryHeap::new();
        let mut parents: SlotMap<slotmap::DefaultKey, (slotmap::DefaultKey, Cfg)>  = SlotMap::new();
        {
        let start_key = parents.insert_with_key(|k| (k, Cfg(start.clone(), 0, 0)));
        to_see.push( SmallestCostHolder{
            estimated_cost: 0,
            cost: 0,
            key: start_key,
        });
        }
        while let Some(SmallestCostHolder{cost, key, ..}) = to_see.pop() {
            let successors = {
                let &(final_key, Cfg(node, _interval, arrival_time)) = parents.get(key).unwrap(); // Cannot fail
                if success(&node) {
                    let path = reverse_path(&parents, |&key| key, final_key);
                    return Some((path, arrival_time));
                }
                if cost > arrival_time {
                    continue;
                }
                successors(&node)
            };
            for (successor,interval, move_cost) in successors {

            }
        }
        None
    }


    #[allow(clippy::needless_collect)]
    fn reverse_path<K, V, F>(parents: &SlotMap<K, (K, V)>, mut parent: F, start: K ) -> Vec<V>
    where
        K: slotmap::Key,
        F: FnMut(&K) -> K,
        V: Clone,
    {
        let mut i = start;
        let path = std::iter::from_fn(|| {
            parents.get(i).map(|(key, cfg)| {
                i = parent(key);
                cfg
            })
        })
        .collect::<Vec<&V>>();
        // Collecting the going through the vector is needed to revert the path because the
        // unfold iterator is not double-ended due to its iterative nature.
        path.into_iter().rev().cloned().collect()
    }

    struct SmallestCostHolder<C, K> {
        estimated_cost: C,
        cost: C,
        key: K,
    }

    use std::cmp::Ordering;

    impl<C: PartialEq, K> PartialEq for SmallestCostHolder<C, K> {
        fn eq(&self, other: &Self) -> bool {
            self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
        }
    }

    impl<C: PartialEq, K> Eq for SmallestCostHolder<C, K> {}

    impl<C: Ord, K> PartialOrd for SmallestCostHolder<C, K> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<C: Ord, K> Ord for SmallestCostHolder<C, K> {
        fn cmp(&self, other: &Self) -> Ordering {
            match other.estimated_cost.cmp(&self.estimated_cost) {
                Ordering::Equal => self.cost.cmp(&other.cost),
                s => s,
            }
        }
}

}


pub mod map_loader;
pub mod plot_tools;
pub mod search_algorithms;

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

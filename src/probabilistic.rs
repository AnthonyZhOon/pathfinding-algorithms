use ndarray::{Array, Array2};

use pathfinding::prelude::Grid;

pub fn blur_to_prob(grid: &Grid) -> Array2<f64> {
    // Consider multiple iterations of blurring
    let arr = {
        let mut tmp = vec![0.; grid.width * grid.height];
        grid.iter().for_each(|(x, y)| tmp[x + y * grid.width] = 1.);
        tmp
    };
    let base = Array::from_shape_vec((grid.height, grid.width), arr).unwrap();
    convolve_2d(&base, &Array::from_elem((3, 3), 1. / 9.))
}

pub fn sample_prob_grid(probs: &Array2<f64>) -> Grid {
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

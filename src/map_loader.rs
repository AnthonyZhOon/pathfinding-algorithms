use pathfinding::grid::Grid;
use std::fs::File;
use std::io::prelude::*;

fn read_line(lines: &mut core::str::Split<'_, char>) -> String {
    lines.next().unwrap().to_string()
}

pub fn read_map(path: &str) -> Grid {
    // Expects to read a .map file
    let mut text = String::new();
    let mut file = File::open(path).unwrap_or_else(|_e| panic!("Failed to open file at {path}"));
    file.read_to_string(&mut text)
        .expect("Failed to read file to string");
    let mut file = text.split('\n');

    let _map_type = read_line(&mut file);
    let height = read_line(&mut file)
        .split_whitespace()
        .last()
        .unwrap()
        .parse::<usize>()
        .expect("Failed to parse height");
    let width = read_line(&mut file)
        .split_whitespace()
        .last()
        .unwrap()
        .parse::<usize>()
        .expect("Failed to parse width");
    let _name = read_line(&mut file);
    let mut map = Grid::new(width, height);
    let mut idx = 0;
    for (y, line) in file.enumerate() {
        println!("{:?}", line);
        for (x, chr) in line.chars().enumerate() {
            // print!{"{char}"};
            // println!("{x}, {y}");
            // println!("{}", map.is_inside((x, y)));
            match chr {
                '@' | 'T' => {
                    map.add_vertex((x, y));
                    idx += 1
                }
                '.' => {}
                _ => {}
            }
        }
        // println!();
    }
    println!("{idx}, {}", map.vertices_len());

    map.invert();
    map.enable_diagonal_mode();
    for y in 0..map.height {
        for x in 0..map.width {
            if map.has_vertex((x, y)) {
                print!(".");
            } else {
                print!("@");
            }
        }
        println!();
    }
    map
}

// Perform Depth-First Search using recursive calls
use std::hash::Hash;
use std::collections::HashSet;

pub fn dfs<N, FN, FS, IN>(
    start: N,
    mut neighbours: FN,
    mut success: FS,
) -> Option<Vec<N>>
where
    N: Eq + Hash + Clone,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = N>,
    FS: FnMut(&N) -> bool,
    {
        let mut explored = HashSet::new();
        explored.insert(start.clone());
        let mut path = vec![start];
        search(&mut neighbours, &mut success, &mut path, &mut explored).then_some(path)
    }
fn search<N, FN, FS, IN>(
    neighbours: &mut FN,
    success: &mut FS,
    path: &mut Vec<N>,
    explored: &mut HashSet<N>,
) -> bool
where 
    N: Eq + Hash + Clone,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = N>,
    FS: FnMut(&N) -> bool,
    {
        if success(path.last().unwrap()) { return true}
        for next in neighbours(path.last().unwrap()){
            if !explored.contains(&next) {
                explored.insert(next.clone());
                path.push(next);
                if search(neighbours, success, path, explored) {
                    return true
                }
                path.pop();
            }
        }
        false
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(dfs(&1, |x| vec![*x], |_x| false), None)
    }

    #[test]
    fn test_linear() {
        assert_eq!(dfs(1usize, |x| vec![*x+1], |&x| x == 10usize), Some((1..=10usize).collect()))
    }

}
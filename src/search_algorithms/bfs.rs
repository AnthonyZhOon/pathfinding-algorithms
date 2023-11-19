use std::collections::{VecDeque, HashMap};
use std::hash::Hash;


pub fn bfs<N, FN, FS, IN>(
    start: N,
    successors: &mut FN,
    success: &mut FS,
) -> Option<Vec<N>>
where
    N: Eq + Hash + Clone,
    FN: FnMut(&N)-> IN,
    IN: IntoIterator<Item = N>,
    FS: FnMut(&N) -> bool,
    {
        bfs_core(&start, successors, success, true)
    }

fn bfs_core<N, FN, FS, IN>(
    start: &N,
    successors: &mut FN,
    success: &mut FS,
    check_first: bool,
) -> Option<Vec<N>>
where
    N: Eq + Hash + Clone,
    FN: FnMut(&N)-> IN,
    IN: IntoIterator<Item = N>,
    FS: FnMut(&N) -> bool,
    {
    if check_first && success(start) {
        return Some(vec![]);
    }
    let mut frontier = VecDeque::new();
    let mut explored = HashMap::new();
    explored.insert(start.clone(), None);
    frontier.push_back(start.clone());
    while let Some(curr) = frontier.pop_front() {
        if success(&curr) {return trace_path(curr, explored, start)}
        for next in successors(&curr) {
            if !explored.contains_key(&next){
                explored.insert(next.clone(), Some(curr.clone()));
                frontier.push_back(next);
            }
        }
    }
    None
    }

fn trace_path<N: Eq + Hash + Clone>(end: N, explored: HashMap<N,Option<N>>, start: &N) -> Option<Vec<N>> {
    let mut path = vec![end.clone()];
    let mut curr = &end;
    while let Some(node) = explored.get(curr).unwrap() {
        path.push(node.clone());
        curr = node;
        if *node == *start {path.reverse(); return Some(path)}
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(bfs(1,  &mut |x| vec![*x], &mut |_x| false), None)
    }

    #[test]
    fn test_start_is_goal() {
        assert_eq!(bfs(1,  &mut |x| vec![*x + 1], &mut |&x| x == 1), Some(vec![]))
    }

    #[test]
    fn test_linear() {
        assert_eq!(bfs(1usize, &mut |x| vec![*x+1],  &mut |&x| x == 10usize), Some((1..=10usize).collect()))
    }

    #[test]
    fn test_quadratic() {
        assert_eq!(bfs(1, &mut |&x| vec![x*x, x+1],  &mut |&x| x == 100usize), Some(vec![1, 2, 3, 9, 10, 100]))
    }

}
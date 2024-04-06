pub mod bfs {
    use std::collections::{HashMap, VecDeque};
    use std::hash::Hash;

    pub fn bfs<N, FN, FS, IN>(start: N, successors: &mut FN, success: &mut FS) -> Option<Vec<N>>
    where
        N: Eq + Hash + Clone,
        FN: FnMut(&N) -> IN,
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
        FN: FnMut(&N) -> IN,
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
            if success(&curr) {
                return trace_path(curr, explored, start);
            }
            for next in successors(&curr) {
                if !explored.contains_key(&next) {
                    explored.insert(next.clone(), Some(curr.clone()));
                    frontier.push_back(next);
                }
            }
        }
        None
    }

    fn trace_path<N: Eq + Hash + Clone>(
        end: N,
        explored: HashMap<N, Option<N>>,
        start: &N,
    ) -> Option<Vec<N>> {
        let mut path = vec![end.clone()];
        let mut curr = &end;
        while let Some(node) = explored.get(curr).unwrap() {
            path.push(node.clone());
            curr = node;
            if *node == *start {
                path.reverse();
                return Some(path);
            }
        }
        None
    }

    #[cfg(test)]
    mod tests_bfs {
        use super::*;

        #[test]
        fn test_empty() {
            assert_eq!(bfs(1, &mut |x| vec![*x], &mut |_x| false), None)
        }

        #[test]
        fn test_start_is_goal() {
            assert_eq!(
                bfs(1, &mut |x| vec![*x + 1], &mut |&x| x == 1),
                Some(vec![])
            )
        }

        #[test]
        fn test_linear() {
            assert_eq!(
                bfs(1usize, &mut |x| vec![*x + 1], &mut |&x| x == 10usize),
                Some((1..=10usize).collect())
            )
        }

        #[test]
        fn test_quadratic() {
            assert_eq!(
                bfs(1, &mut |&x| vec![x * x, x + 1], &mut |&x| x == 100usize),
                Some(vec![1, 2, 3, 9, 10, 100])
            )
        }
    }
}

pub mod dfs {
    use std::collections::HashSet;
    use std::hash::Hash;

    pub fn dfs<N, FN, FS, IN>(start: N, mut neighbours: FN, mut success: FS) -> Option<Vec<N>>
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
        if success(path.last().unwrap()) {
            return true;
        }
        for next in neighbours(path.last().unwrap()) {
            if !explored.contains(&next) {
                explored.insert(next.clone());
                path.push(next);
                if search(neighbours, success, path, explored) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }

    #[cfg(test)]
    mod tests_dfs {
        use super::*;

        #[test]
        fn test_empty() {
            assert_eq!(dfs(&1, |x| vec![*x], |_x| false), None)
        }

        #[test]
        fn test_linear() {
            assert_eq!(
                dfs(1usize, |x| vec![*x + 1], |&x| x == 10usize),
                Some((1..=10usize).collect())
            )
        }
    }
}

pub mod djikstra {
    use std::cmp::Ordering;
    use std::collections::hash_map::Entry;
    use std::collections::{BinaryHeap, HashMap};
    use std::hash::Hash;
    use std::usize;

    struct SmallestItem<C, N> {
        item: N,
        cost: C,
    }

    impl<C: Ord, N> Ord for SmallestItem<C, N> {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.cost.cmp(&other.cost) {
                Ordering::Equal => Ordering::Equal,
                Ordering::Greater => Ordering::Less,
                Ordering::Less => Ordering::Greater,
            }
        }
    }

    impl<C: Ord, N> PartialOrd for SmallestItem<C, N> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<C: Eq, N> PartialEq for SmallestItem<C, N> {
        fn eq(&self, other: &Self) -> bool {
            self.cost == other.cost
        }
    }

    impl<C: Eq + Ord, N> Eq for SmallestItem<C, N> {}

    // Creating a map from Node to (parent_index, cost) is the goal of djikstra
    // To map from node I need to hash?, But to index the tree I need a treemap
    // Can it just be a hashmap?

    pub fn djikstra<N, FN, FS, IN>(
        start: N,
        mut successors: FN,
        mut success: FS,
    ) -> Option<(Vec<N>, usize)>
    where
        N: Eq + Clone + Hash,
        FN: FnMut(&N) -> IN,
        IN: IntoIterator<Item = (N, usize)>,
        FS: FnMut(&N) -> bool,
    {
        let (parents, target_reached) = djikstra_core(start, &mut successors, &mut success);
        reverse_path(parents, target_reached)
    }

    fn djikstra_core<N, FN, FS, IN>(
        start: N,
        successors: &mut FN,
        success: &mut FS,
    ) -> (HashMap<N, (Option<N>, usize)>, Option<N>)
    where
        N: Eq + Clone + Hash,
        FN: FnMut(&N) -> IN,
        IN: IntoIterator<Item = (N, usize)>,
        FS: FnMut(&N) -> bool,
    {
        let mut results: HashMap<N, (Option<N>, usize)> = HashMap::new();
        let mut to_visit: BinaryHeap<SmallestItem<usize, N>> = BinaryHeap::new();
        to_visit.push(SmallestItem {
            cost: 0,
            item: start.clone(),
        });
        let mut target_reached = None;
        results.insert(start.clone(), (None, 0));
        while let Some(SmallestItem {
            cost,
            item: current,
        }) = to_visit.pop()
        {
            let successors = {
                if success(&current) {
                    target_reached = Some(current);
                    break;
                }
                successors(&current)
            };

            for (neighbour, move_cost) in successors {
                let new_cost = move_cost + cost;

                let e = results.entry(neighbour.clone());
                match e {
                    Entry::Vacant(e) => {
                        e.insert((Some(current.clone()), new_cost));
                    }
                    Entry::Occupied(mut e) => {
                        if e.get().1 > new_cost {
                            e.insert((Some(current.clone()), new_cost));
                        }
                    }
                }
                to_visit.push(SmallestItem {
                    item: neighbour,
                    cost: new_cost,
                })
            }
        }
        (results, target_reached)
    }

    fn reverse_path<N>(
        parents: HashMap<N, (Option<N>, usize)>,
        target_reached: Option<N>,
    ) -> Option<(Vec<N>, usize)>
    where
        N: Eq + Clone + Hash,
    {
        target_reached.as_ref()?; // Return None if target is unreachable
        let mut reverse_path = vec![target_reached.unwrap()];
        let (_, total_cost) = parents.get(reverse_path.first().unwrap()).unwrap();
        while let Some((Some(parent), _)) = parents.get(reverse_path.last().unwrap()) {
            reverse_path.push(parent.clone());
        }
        reverse_path.reverse();
        Some((reverse_path, *total_cost))
    }

    #[cfg(test)]
    mod tests_djikstra {

        use super::*;

        #[test]
        fn test_empty() {
            assert!(djikstra(1, |_x| vec![], |_x| false).is_none());
        }

        #[test]
        fn test_start_is_goal() {
            assert!(matches!(
                djikstra(1, |&x| vec![(x + 1, x)], |&x| x == 1),
                Some((_, 0))
            ));
        }

        #[test]
        fn test_triangle_inequality() {
            let results = djikstra(1, |&x| vec![(x + 1, 1), (x + 2, 4)], |&x| x == 3);
            assert!(matches!(results, Some((_, 2))));
            assert_eq!(results.unwrap().0, vec![1, 2, 3])
        }
    }
}

#[allow(unused)]
mod safe_interval {
    use crate::map_gen::random_grid;
    use pathfinding::prelude::Grid;
    use std::collections::HashMap;
    type Pos = (usize, usize);
    // Generate a grid, apply the dynamic obstacles to create safe intervals
    // Run djikstra_all for a heuristic
    // Create the neighbours function
    struct PosWithIntervals {
        position: Pos,
        safe_intervals: Vec<(usize, usize)>,
    }

    struct NodeWithInterval(Pos, usize);

    fn prepare_grid() {
        let grid: Grid = random_grid(300, 300, 10);
        let mut interval_grid: HashMap<Pos, PosWithIntervals> = HashMap::new();
        for vertex in grid.iter() {
            interval_grid.insert(
                vertex,
                PosWithIntervals {
                    position: vertex,
                    safe_intervals: vec![(0, 99)],
                },
            );
        }
    }
}

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap,};
use std::hash::Hash;
use std::usize;


struct SmallestItem<C, N> {
    item: N,
    cost: C
}

impl<C: Ord, N> Ord for SmallestItem<C, N> {
   fn cmp(&self, other: &Self) -> Ordering {
       match self.cost.cmp(&other.cost) {
            Ordering::Equal => Ordering::Equal,
            Ordering::Greater => Ordering::Less,
            Ordering::Less => Ordering::Greater
       }
   }
}

impl<C: Ord, N> PartialOrd for SmallestItem<C, N>{
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
    mut success: FS
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
    success: &mut FS
) -> (HashMap<N, (Option<N>, usize)>, Option<N>)
where
    N: Eq + Clone + Hash,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = (N, usize)>,
    FS: FnMut(&N) -> bool,
 {
    let mut results: HashMap<N, (Option<N>, usize)> = HashMap::new();
    let mut to_visit:BinaryHeap<SmallestItem<usize, N>> = BinaryHeap::new();
    to_visit.push(SmallestItem{
        cost: 0,
        item: start.clone()
    });
    let mut target_reached = None;
    results.insert(start.clone(), (None, 0));
    while let Some(SmallestItem{cost, item: current }) = to_visit.pop() {
        let successors = {
            if success(&current) {
                target_reached = Some(current);
                break 
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
            to_visit.push(SmallestItem { item: neighbour, cost: new_cost })
        }
    }
    (results, target_reached)

}

fn reverse_path<N>(parents: HashMap<N, (Option<N>, usize)>, target_reached: Option<N>) -> Option<(Vec<N>, usize)>
where
    N: Eq + Clone + Hash
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
mod tests {

    use super::*;

    #[test]
    fn test_empty() {
        assert!(djikstra(1, |_x| vec![], |_x| false).is_none());
    }

    #[test]
    fn test_start_is_goal() {
        assert!(matches!(djikstra(1, |&x| vec![(x + 1, x)], |&x| x == 1), Some((_, 0))));
    }

    #[test]
    fn test_triangle_inequality() {
        let results = djikstra(1, |&x| vec![(x + 1, 1), (x + 2, 4)], |&x| x == 3);
        assert!(matches!(results, Some((_, 2))));
        assert_eq!(results.unwrap().0, vec![1,2,3])
    }

}



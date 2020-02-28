//! An implementation of disjoint set forests for union find.

/// Data structure for efficient union find.
pub struct DisjointSetForest {
    /// Number of forest elements.
    count: usize,
    /// parent[i] is the index of the parent
    /// of the element with index i. If parent[i] == i
    /// then i is a root.
    parent: Vec<usize>,
    /// tree_size[i] is the size of the tree rooted at i.
    tree_size: Vec<usize>,
}

impl DisjointSetForest {
    /// Constructs forest of singletons with count elements.
    pub fn new(count: usize) -> DisjointSetForest {
        let parent: Vec<usize> = (0..count).collect();
        let tree_size = vec![1 as usize; count];
        DisjointSetForest {
            count,
            parent,
            tree_size,
        }
    }

    /// Returns the number of trees in the forest.
    pub fn num_trees(&self) -> usize {
        self.parent
            .iter()
            .enumerate()
            .fold(0, |acc, (i, p)| acc + if i == *p { 1 } else { 0 })
    }

    /// Returns index of the root of the tree containing i.
    /// Needs mutable reference to self for path compression.
    pub fn root(&mut self, i: usize) -> usize {
        assert!(i < self.count);
        let mut j = i;
        loop {
            unsafe {
                let p = *self.parent.get_unchecked(j);
                *self.parent.get_unchecked_mut(j) = *self.parent.get_unchecked(p);
                if j == p {
                    break;
                }
                j = p;
            }
        }
        j
    }

    /// Returns true if i and j are in the same tree.
    /// Need mutable reference to self for path compression.
    pub fn find(&mut self, i: usize, j: usize) -> bool {
        assert!(i < self.count && j < self.count);
        self.root(i) == self.root(j)
    }

    /// Unions the trees containing i and j.
    pub fn union(&mut self, i: usize, j: usize) {
        assert!(i < self.count && j < self.count);
        let p = self.root(i);
        let q = self.root(j);
        if p == q {
            return;
        }
        unsafe {
            let p_size = *self.tree_size.get_unchecked(p);
            let q_size = *self.tree_size.get_unchecked(q);
            if p_size < q_size {
                *self.parent.get_unchecked_mut(p) = q;
                *self.tree_size.get_unchecked_mut(q) = p_size + q_size;
            } else {
                *self.parent.get_unchecked_mut(q) = p;
                *self.tree_size.get_unchecked_mut(p) = p_size + q_size;
            }
        }
    }

    /// Returns the elements of each tree.
    pub fn trees(&mut self) -> Vec<Vec<usize>> {
        use std::collections::HashMap;

        // Maps a tree root to the index of the set
        // containing its children
        let mut root_sets: HashMap<usize, usize> = HashMap::new();

        let mut sets: Vec<Vec<usize>> = vec![];
        for i in 0..self.count {
            let root = self.root(i);
            match root_sets.get(&root).cloned() {
                Some(set_idx) => {
                    sets[set_idx].push(i);
                }
                None => {
                    let idx = sets.len();
                    let set = vec![i];
                    sets.push(set);
                    root_sets.insert(root, idx);
                }
            }
        }
        sets
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSetForest;
    use ::test;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Uniform};

    #[test]
    fn test_trees() {
        //    3         4
        //    |        /  \
        //    1       5    7
        //   /  \     |
        //  0    2    6
        #[rustfmt::skip]
        let mut forest = DisjointSetForest {
            count: 8,
            // element:     0, 1, 2, 3, 4, 5, 6, 7
            parent:    vec![1, 3, 1, 3, 4, 4, 5, 4],
            tree_size: vec![1, 3, 1, 4, 4, 2, 1, 1],
        };

        assert_eq!(forest.trees(), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    }

    #[test]
    fn test_union_find_sequence() {
        let mut forest = DisjointSetForest::new(6);
        // 0  1  2  3  4  5

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(forest.num_trees(), 6);

        forest.union(0, 4);
        // 0  1  2  3  5
        // |
        // 4

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 3, 0, 5]);
        assert_eq!(forest.num_trees(), 5);

        forest.union(1, 3);
        // 0  1  2  5
        // |  |
        // 4  3

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 1, 0, 5]);
        assert_eq!(forest.num_trees(), 4);

        forest.union(3, 2);
        // 0    1     5
        // |   / \
        // 4  3   2

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 1, 1, 0, 5]);
        assert_eq!(forest.num_trees(), 3);

        forest.union(2, 4);
        //    1     5
        //  / | \
        // 0  3  2
        // |
        // 4

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![1, 1, 1, 1, 0, 5]);
        assert_eq!(forest.num_trees(), 2);
    }

    #[bench]
    fn bench_disjoint_set_forest(b: &mut test::Bencher) {
        let num_nodes = 500;
        let num_edges = 20 * num_nodes;

        let mut rng: StdRng = SeedableRng::seed_from_u64(1);
        let uniform = Uniform::new(0, num_nodes);

        let mut forest = DisjointSetForest::new(num_nodes);
        b.iter(|| {
            let mut count = 0;
            while count < num_edges {
                let u = uniform.sample(&mut rng);
                let v = uniform.sample(&mut rng);
                forest.union(u, v);
                count += 1;
            }
            test::black_box(forest.num_trees());
        });
    }
}

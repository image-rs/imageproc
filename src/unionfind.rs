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
        let mut parent = vec![0 as usize; count];
        for i in 0..count {
            parent[i] = i;
        }
        let tree_size = vec![1 as usize; count];
        DisjointSetForest {
            count: count,
            parent: parent,
            tree_size: tree_size}
    }

    /// Returns index of the root of the tree containing i.
    /// Needs mutable reference to self for path compression.
    pub fn root(&mut self, i: usize) -> usize {
        let mut j = i;
        loop {
            let p = self.parent[j];
            self.parent[j] = self.parent[p];
            if j == p {
                break;
            }
            j = p;
        }
        j
    }

    /// Returns true if i and j are in the same tree.
    /// Need mutable reference to self for path compression.
    pub fn find(&mut self, i: usize, j: usize) -> bool {
        self.root(i) == self.root(j)
    }

    /// Unions the trees containing i and j.
    pub fn union(&mut self, i: usize, j: usize) {
        let p = self.root(i);
        let q = self.root(j);
        if self.tree_size[p] < self.tree_size[q] {
            self.parent[p] = q;
            self.tree_size[q] += self.tree_size[p];
        }
        else {
            self.parent[q] = p;
            self.tree_size[p] += self.tree_size[q];
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
            match root_sets.get(&root).map(|x| *x) {
                Some(set_idx) => {
                    sets[set_idx].push(i);
                },
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
mod test {

    use super::{
        DisjointSetForest
    };

    #[test]
    fn test_trees() {
        //    3         4
        //    |        /  \
        //    1       5    7
        //   /  \     |
        //  0    2    6
        let mut forest = DisjointSetForest {
            count: 8,
            // element:     0, 1, 2, 3, 4, 5, 6, 7
            parent:    vec![1, 3, 1, 3, 4, 4, 5, 4],
            tree_size: vec![1, 3, 1, 4, 4, 2, 1, 1]
        };

        assert_eq!(forest.trees(),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    }

    #[test]
    fn test_union_find_sequence() {

        let mut forest = DisjointSetForest::new(6);
        // 0  1  2  3  4  5

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 3, 4, 5]);

        forest.union(0, 4);
        // 0  1  2  3  5
        // |
        // 4

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 3, 0, 5]);


        forest.union(1, 3);
        // 0  1  2  5
        // |  |
        // 4  3

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 2, 1, 0, 5]);

        forest.union(3, 2);
        // 0    1     5
        // |   / \
        // 4  3   2

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![0, 1, 1, 1, 0, 5]);

        forest.union(2, 4);
        //    1     5
        //  / | \
        // 0  3  2
        // |
        // 4

        //                             0, 1, 2, 3, 4, 5
        assert_eq!(forest.parent, vec![1, 1, 1, 1, 0, 5]);
    }

    // TODO: write a benchmark
}

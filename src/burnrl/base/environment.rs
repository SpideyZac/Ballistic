use std::fmt::Debug;

use crate::burnrl::base::{Action, ElemType, Snapshot, State};

pub trait Environment: Debug {
    type StateType: State;
    type ActionType: Action;
    type RewardType: Debug + Clone + Into<ElemType>;

    const MAX_STEPS: usize = usize::MAX;

    fn new(visualized: bool) -> Self;

    fn state(&self) -> Self::StateType;

    fn reset(&mut self) -> Snapshot<Self>;

    fn render(&self);

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self>;
}

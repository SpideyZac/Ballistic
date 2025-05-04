use crate::burnrl::base::environment::Environment;

pub trait Agent<E: Environment> {
    fn react(&self, state: &E::StateType) -> Option<E::ActionType>;
}

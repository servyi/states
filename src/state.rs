use anyhow::Result;
use async_trait::async_trait;
use futures::future::try_join_all;

pub enum Step<K> {
    Continue(Box<dyn State<K>>),
    Done(K),
}

#[async_trait]
pub trait State<K>: Send + Sync + 'static {
    async fn execute(self: Box<Self>) -> Result<Step<K>>;
}

#[async_trait]
pub trait IOState<K>: Send + Sync + 'static {
    type Output: Send + Sync + 'static;

    async fn perform_io(&self) -> Result<Self::Output>;
    async fn transition(self: Box<Self>, result: Self::Output) -> Result<Step<K>>;
}

#[async_trait]
pub trait ParallelState<K, I>: Send + Sync + 'static
where
    I: Send + Sync + 'static,
{
    async fn split(&self) -> Result<Vec<Box<dyn State<I>>>>;
    async fn collect(self: Box<Self>, results: Vec<I>) -> Result<Step<K>>;
}

pub async fn run_to_completion<I>(mut state: Box<dyn State<I>>) -> Result<I>
where
    I: Send + Sync + 'static,
{
    loop {
        match state.execute().await? {
            Step::Continue(next) => state = next,
            Step::Done(result) => return Ok(result),
        }
    }
}

pub async fn execute_io<K, S>(state: Box<S>) -> Result<Step<K>>
where
    S: IOState<K>,
{
    let output = state.perform_io().await?;
    state.transition(output).await
}

pub async fn execute_parallel<K, I, S>(state: Box<S>) -> Result<Step<K>>
where
    K: Send + Sync + 'static,
    I: Send + Sync + 'static,
    S: ParallelState<K, I>,
{
    let sub_states = state.split().await?;
    let futures: Vec<_> = sub_states
        .into_iter()
        .map(|s| run_to_completion(s))
        .collect();
    let results = try_join_all(futures).await?;
    state.collect(results).await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Increment(u32);
    #[async_trait]
    impl State<u32> for Increment {
        async fn execute(self: Box<Self>) -> Result<Step<u32>> {
            if self.0 > 0 {
                Ok(Step::Continue(Box::new(Increment(self.0 - 1))))
            } else {
                Ok(Step::Done(42))
            }
        }
    }

    #[test]
    fn test_run_to_completion_linear() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(run_to_completion(Box::new(Increment(3))));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_run_to_completion_immediate() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(run_to_completion(Box::new(Increment(0))));
        assert_eq!(result.unwrap(), 42);
    }

    struct FailState;
    #[async_trait]
    impl State<()> for FailState {
        async fn execute(self: Box<Self>) -> Result<Step<()>> {
            anyhow::bail!("intentional failure")
        }
    }

    #[test]
    fn test_state_error_propagates() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(run_to_completion::<()>(Box::new(FailState)));
        assert!(result.is_err());
    }
}
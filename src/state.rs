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
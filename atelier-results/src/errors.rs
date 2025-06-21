use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum LevelError {
    // Level not found
    #[error("Level not found")]
    LevelNotFound,

    // Level info not available
    #[error("Level info not available")]
    LevelInfoNotAvailable,

    // Level deletion not successful
    #[error("Level deletion not successful")]
    LevelDeletionFailed,

    // Level modification not succesful
    #[error("Level modification not successful")]
    LevelModificationFailed,

    // Level insertion not successful
    #[error("Level insertion not successful")]
    LevelInsertionFailed,
}

#[derive(Error, Debug)]
pub enum OrderError {
    // Order not found
    #[error("Order not found")]
    OrderNotFound,

    // Order info not available
    #[error("Order info not available")]
    OrderInfoNotAvailable,

    // Order deletion not successful
    #[error("Order deletion not successful")]
    OrderDeletionFailed,

    // Order modification not successful
    #[error("Order modification not successful")]
    OrderModificationFailed,

    // Order insertion not succesful
    #[error("Order insertion not successful")]
    OrderInsertionFailed,
}

#[derive(Error, Debug)]
pub enum GeneratorError {
    // Undefined Generator Error
    #[error("The Generator presented an Undefined Error")]
    GeneratorUndefinedError,

    // Not a valid number on the input
    #[error("The Generator did not recived a valid number")]
    GeneratorInputTypeFailure,

    // Not a valid number on the output
    #[error("The Generator did not produced a valid number")]
    GeneratorOutputTypeFailure,
}

#[derive(Error, Debug)]
pub enum EventError {
    // Event generator failed
    #[error("The Event Generator function failed")]
    EventFailure,
}

#[derive(Error, Debug)]
pub enum SynthetizerError {
    // Progression Generation Error
    #[error("The progression generation was unsuccessful {0}")]
    GenerationError(String),
}


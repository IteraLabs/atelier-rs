#[cfg(test)]

// -- ----------------------------------------------------------------- TESTS UTILS -- //
// -- ----------------------------------------------------------------- ----------- -- //

mod test_utils {

    use tch::{Kind, Tensor};

    pub fn series() -> (Tensor, Tensor) {
        let y_true = Tensor::from_slice(&vec![1.0, 0.0, 0.0, 1.0]).to_kind(Kind::Float);
        let y_hat = Tensor::from_slice(&vec![1.0, 1.0, 0.0, 0.0]).to_kind(Kind::Float);

        (y_true, y_hat)
    }
}

mod tests {

    // --------------------------------------------------------------- OUTPUT VALUE -- //
    use crate::test_utils::series;
    use atelier_dcml::functions;
    use tch::Tensor;

    #[test]
    fn test_loss_function() {
        let (y_true, y_hat) = series();
        let loss = functions::CrossEntropy::new()
            .id(&"CrossEntropy".to_string())
            .build();
        let computed_loss = loss.unwrap().compute_loss(&y_hat, &y_true);

        assert_eq!(computed_loss, Tensor::from(0.7532044649124146));
    }
}

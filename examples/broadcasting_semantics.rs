use candle_core::{DType, Device, Result, Tensor};

/// This example demonstrates broadcasting semantics in the Candle framework.
///
/// Broadcasting allows operations between tensors of different shapes by following these rules:
/// 1. Dimensions are matched from right to left
/// 2. Each dimension pair must either:
///    - Be exactly equal
///    - Have one dimension = 1 (which will be broadcast to match the larger dim)
///    - Have one dimension missing (treated as 1 and broadcast)
///
/// For example, broadcasting shapes (5,3,4,1) + (3,1,1):
/// - Right-to-left: 1==1, 4>1 (broadcast), 3==3, 5 has no pair (broadcast)
/// - Result shape: (5,3,4,1)
fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("Broadcasting Semantics in Candle");
    println!("================================\n");

    // Example 1: Same shapes (trivial broadcasting)
    // When shapes are identical, broadcasting is straightforward -
    // each element pairs with its corresponding element
    let x = Tensor::ones((5, 7, 3), DType::F64, &device)?;
    let y = Tensor::ones((5, 7, 3), DType::F64, &device)?;
    println!("Same shapes: {:?} + {:?}", x.shape(), y.shape());
    let z = x.broadcast_add(&y)?;
    println!("Result shape: {:?}\n", z.shape());

    // Example 1.5: Invalid broadcasting with empty tensor
    // Cannot broadcast when one tensor has 0 dimensions
    let x = Tensor::ones(0, DType::F64, &device)?;
    let y = Tensor::ones((2, 2), DType::F64, &device)?;
    println!("Can't broadcast: {:?} + {:?}", x.shape(), y.shape());
    match x.broadcast_add(&y) {
        Ok(_) => println!("Succeeded (unexpected)"),
        Err(e) => println!("Failed as expected: {}", e),
    }

    // Example 2: Broadcasting with trailing dimensions
    // Demonstrates right-to-left matching with missing dimensions:
    // x: (5,3,4,1) - from right to left:
    // y: (  3,1,1) - 1==1, 4>1 (broadcast), 3==3, 5 (broadcast)
    let x = Tensor::ones((5, 3, 4, 1), DType::F64, &device)?;
    let y = Tensor::ones((3, 1, 1), DType::F64, &device)?;
    println!(
        "Broadcasting trailing dims: {:?} + {:?}",
        x.shape(),
        y.shape()
    );
    let z = x.broadcast_add(&y)?;
    println!("Result shape: {:?}\n", z.shape());

    // Example 3: Scalar broadcasting
    // A (1,1) tensor acts like a scalar and broadcasts to any shape
    let x = Tensor::ones((3, 4), DType::F64, &device)?;
    let y = Tensor::ones((1, 1), DType::F64, &device)?;
    println!("Scalar broadcasting: {:?} + {:?}", x.shape(), y.shape());
    let z = x.broadcast_add(&y)?;
    println!("Result shape: {:?}\n", z.shape());

    // Example 4: Broadcasting with ones
    // When a dimension is 1, it's stretched to match the corresponding dimension
    // Here (2,1,3) + (2,4,3) -> (2,4,3)
    let x = Tensor::ones((2, 1, 3), DType::F64, &device)?;
    let y = Tensor::ones((2, 4, 3), DType::F64, &device)?;
    println!("Broadcasting with ones: {:?} + {:?}", x.shape(), y.shape());
    let z = x.broadcast_add(&y)?;
    println!("Result shape: {:?}\n", z.shape());

    // Example 5: Invalid broadcasting
    // Shapes (2,3) and (2,2) cannot broadcast because:
    // - Right-most dimensions (3 vs 2) neither match nor have a 1
    let x = Tensor::ones((2, 3), DType::F64, &device)?;
    let y = Tensor::ones((2, 2), DType::F64, &device)?;
    println!(
        "Invalid broadcasting attempt: {:?} + {:?}",
        x.shape(),
        y.shape()
    );
    match x.broadcast_add(&y) {
        Ok(_) => println!("Succeeded (unexpected)"),
        Err(e) => println!("Failed as expected: {}", e),
    }

    // Example 6: Complex broadcasting with multiple ones
    // Demonstrates broadcasting across multiple dimensions:
    // (1,5,1,2) + (3,1,4,2) -> (3,5,4,2)
    let x = Tensor::ones((1, 5, 1, 2), DType::F64, &device)?;
    let y = Tensor::ones((3, 1, 4, 2), DType::F64, &device)?;
    println!("Complex broadcasting: {:?} + {:?}", x.shape(), y.shape());
    let z = x.broadcast_add(&y)?;
    println!("Result shape: {:?}\n", z.shape());

    // Example 7: Broadcasting with different operations
    // The same broadcasting rules apply to all operations (mul, sub, div, etc.)
    let x = Tensor::ones((4, 1), DType::F64, &device)?;
    let y = Tensor::ones((1, 3), DType::F64, &device)?;
    println!("Broadcasting operations: {:?} × {:?}", x.shape(), y.shape());
    let z_mul = x.broadcast_mul(&y)?;
    let z_div = x.broadcast_div(&y)?;
    println!("Multiplication result: {:?}", z_mul.shape());
    println!("Division result: {:?}\n", z_div.shape());

    Ok(())
}

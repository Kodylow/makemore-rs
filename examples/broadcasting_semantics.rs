use candle_core::{DType, Device, Result, Tensor};

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
///
/// Broadcasting Mechanics:
/// 1. Right Alignment:
///    Shapes are aligned on their right edges
///    (5,3,4,1)     ->  5  3  4  1
///    (  3,1,1)     ->     3  1  1
///
///    Which becomes broadcastable by:
///    (5,3,4,1)     ->  5  3  4  1
///    (  3,1,1)     ->  1  3  1  1
///
/// 2. Dimension Expansion:
///    Size-1 dims are stretched to match larger dims
///    (5,3,4,1)     ->  5  3  4  1
///    (3,1,1)       ->  -  3  4  1  
///
/// 3. Missing Dimension Handling:
///    Missing dims (marked -) are treated as 1 and broadcast
///    Final: (5,3,4,1)
///
/// Keepdim is a boolean flag that determines whether the reduced dimensions should be preserved as size 1.
/// This is useful in situations where you want to maintain the shape of the tensor after an operation and
/// will avoid some of the common pitfalls of broadcasting listed below.
///
/// Common Pitfalls:
/// 1. Silent Dimension Addition:
///    When broadcasting adds dimensions, it prepends them to the left:
///    a = (3,4)          ->  1  3  4    // Silent 1 added
///    b = (1,3,4)        ->  1  3  4
///    result = (1,3,4)   // Not (3,4)!
///
/// 2. Unexpected Row/Column Operations:
///    mean(x, dim=0) on (3,4):
///    - Returns (4,)     // Column means
///    - To get row means, use dim=1 -> (3,)
///    - Or transpose first: x.T.mean(dim=1)
///
/// 3. Broadcasting vs Batching:
///    (32,1,28,28) + (1,3,28,28):
///    - Looks like batch_size=32, channels=3
///    - Actually broadcasts to (32,3,28,28)
///    - May silently compute wrong result
///
/// Best Practices:
/// - Always check output shapes
/// - Use shape assertions in critical code
/// - Consider explicit reshapes over implicit broadcasting
/// - Use keepdim=true to maintain dimensionality
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

    // Example 8: Keepdim Basics
    // keepdim=true preserves reduced dimensions as size 1
    let x = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3))?;
    println!("\nKeepdim Operations");
    println!("Original shape: {:?}", x.shape());
    println!("x = \n{}", x);

    // Without keepdim - dimension is removed
    let sum_no_keepdim = x.sum(1)?;
    println!("\nSum without keepdim (dim=1): \n{}", sum_no_keepdim);
    println!("Shape: {:?}", sum_no_keepdim.shape());

    // With keepdim - dimension is preserved as size 1
    let sum_keepdim = x.sum_keepdim(1)?;
    println!("\nSum with keepdim (dim=1): \n{}", sum_keepdim);
    println!("Shape: {:?}", sum_keepdim.shape());

    // Example 9: Multiple Keepdim Operations
    let x = Tensor::arange(0f32, 24f32, &device)?.reshape((2, 3, 4))?;
    println!("\nMultiple Keepdim Operations");
    println!("Original shape: {:?}", x.shape());

    // Chain of keepdim operations preserves broadcasting ability
    let result = x
        .max_keepdim(1)? // Reduces dim 1, keeps shape (2, 1, 4)
        .mean_keepdim(2)?; // Reduces dim 2, keeps shape (2, 1, 1)
    println!(
        "After max_keepdim(1) -> mean_keepdim(2): {:?}",
        result.shape()
    );

    // Example 10: Broadcasting with Keepdim
    println!("\nBroadcasting with Keepdim");
    let x = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    println!("Original: {:?}", x.shape());

    // Create (3,1) tensor using keepdim
    let row_means = x.mean_keepdim(1)?;
    println!("Row means (keepdim): {:?}", row_means.shape());

    // Broadcasting (3,4) - (3,1) -> (3,4)
    let centered = x.broadcast_sub(&row_means)?;
    println!("After broadcasting subtraction: {:?}", centered.shape());

    // Example 11: Complex Broadcasting + Keepdim
    println!("\nComplex Broadcasting + Keepdim");
    let x = Tensor::arange(0f32, 24f32, &device)?.reshape((2, 3, 4))?;

    // Create tensors with different keepdim axes
    let a = x.max_keepdim(0)?; // Shape: (1, 3, 4)
    let b = x.mean_keepdim(1)?; // Shape: (2, 1, 4)
    let c = x.sum_keepdim(2)?; // Shape: (2, 3, 1)

    // Broadcast all three together
    let result = a.broadcast_add(&b)?.broadcast_add(&c)?;
    println!("Broadcasting shapes:");
    println!("  {:?} (max_keepdim(0))", a.shape());
    println!("  {:?} (mean_keepdim(1))", b.shape());
    println!("  {:?} (sum_keepdim(2))", c.shape());
    println!("Result: {:?}", result.shape());

    Ok(())
}

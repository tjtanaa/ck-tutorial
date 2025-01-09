# What is the difference between datatype?


Float stored in 8 bits - ONNX 1.18.0 documentation
The two data types, `Float8_e4m3fnuz` and `Float8_e4m3fn`, are both 8-bit floating-point formats with similar structures but differ in their specific configurations and handling of certain edge cases. Here are the key differences:

### 1. **Bias**
- **Float8_e4m3fnuz**: The bias for the exponent is **8**.
- **Float8_e4m3fn**: The bias for the exponent is **7**.

### 2. **Special Values**
- **Float8_e4m3fnuz**:
  - **No infinities**: This format does not support infinities.
  - **No negative zero**: There is no representation for negative zero.
  - **NaN representation**: NaN is represented when the sign bit is 1 and the rest of the bits are 0s (i.e., `0x80`).

- **Float8_e4m3fn**:
  - **Supports infinities**: This format can represent infinities.
  - **Supports negative zero**: There is a representation for negative zero.
  - **NaN representation**: NaN is represented when all exponent and mantissa bits are set to 1 (i.e., `0x7F`).

### 3. **Denormal Handling**
- **Float8_e4m3fnuz**:
  - Uses a specific denormal mask to handle numbers smaller than the smallest normal number.
  - The denormal mask is calculated as `((127 - 8) + (23 - 3) + 1)`.

- **Float8_e4m3fn**:
  - Also uses a denormal mask but with a different calculation.
  - The denormal mask is calculated as `((127 - 7) + (23 - 3) + 1)`.

### 4. **Maximum Representable Value**
- **Float8_e4m3fnuz**:
  - The maximum value before overflow is `256.0f`.
  - Represented in binary as `1 0000 000` in fp8e4m3fnuz format.

- **Float8_e4m3fn**:
  - The maximum value before overflow is `480.0f`.
  - Represented in binary as `0 1111 111` in fp8e4m3fn format.

### 5. **Implementation Details**
- **Float8_e4m3fnuz**:
  - Based on the paper [https://arxiv.org/pdf/2206.02915.pdf](https://arxiv.org/pdf/2206.02915.pdf).
  - The implementation is derived from the existing `Float8_e4m3fn` implementation.

- **Float8_e4m3fn**:
  - Based on the paper [https://arxiv.org/pdf/2209.05433.pdf](https://arxiv.org/pdf/2209.05433.pdf).
  - Inspired by the Half implementation from `pytorch/c10/util/Half.h`.

### 6. **Conversion and Arithmetic Operations**
- Both formats convert to `float32` for arithmetic operations.
- The conversion functions (`fp8e4m3fnuz_from_fp32_value`, `fp8e4m3fn_from_fp32_value`, etc.) handle the specific rules for each format, including denormalization, rounding, and special values.

### Summary
While both `Float8_e4m3fnuz` and `Float8_e4m3fn` are 8-bit floating-point formats with 1 sign bit, 4 exponent bits, and 3 mantissa bits, they differ in their bias, handling of special values (like infinities, negative zero, and NaN), and the maximum representable value. These differences make them suitable for different use cases where specific precision and range requirements are needed.

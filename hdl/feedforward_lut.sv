// Fixed-point 1-cycle feedback block with explicit two-cycle latency.
module feedforward_lut #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS = 15
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     in_valid,
    input  logic signed [DATA_WIDTH-1:0] in_sample,
    output logic                     out_valid,
    output logic signed [DATA_WIDTH-1:0] out_sample
);
    localparam int LUT_ADDR_BITS = 4;
    localparam int PIPE_LATENCY = 2;

    localparam logic signed [DATA_WIDTH-1:0] LUT[0:15] = '{
        16'sh0000,
        16'sh179F,
        16'sh2DAF,
        16'sh4105,
        16'sh510B,
        16'sh5DB7,
        16'sh675E,
        16'sh6E85,
        16'sh73B4,
        16'sh7768,
        16'sh7A05,
        16'sh7BDB,
        16'sh7D22,
        16'sh7E05,
        16'sh7EA2,
        16'sh7F0F
    };

    logic signed [DATA_WIDTH-1:0] sample_pipe;
    logic sample_valid_pipe;
    logic signed [DATA_WIDTH-1:0] lut_out_pipe;
    logic lut_valid_pipe;

    logic signed [DATA_WIDTH-1:0] abs_sample;
    logic [LUT_ADDR_BITS-1:0] lut_addr;
    logic signed [DATA_WIDTH-1:0] lut_value;

    always_comb begin
        logic [DATA_WIDTH-1:0] abs_candidate;
        abs_candidate = sample_pipe[DATA_WIDTH-1] ? (~sample_pipe + 1'b1) : sample_pipe;
        if (abs_candidate > {1'b0, {DATA_WIDTH-1{1'b1}}}) begin
            abs_candidate = {1'b0, {DATA_WIDTH-1{1'b1}}};
        end
        abs_sample = abs_candidate[DATA_WIDTH-1:0];
        lut_addr = abs_sample[DATA_WIDTH-1 -: LUT_ADDR_BITS];
        lut_value = LUT[lut_addr];
        if (sample_pipe[DATA_WIDTH-1]) begin
            lut_value = -lut_value;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_pipe    <= '0;
            sample_valid_pipe <= 1'b0;
            lut_out_pipe   <= '0;
            lut_valid_pipe <= 1'b0;
            out_sample     <= '0;
            out_valid      <= 1'b0;
        end else begin
            sample_pipe      <= in_sample;
            sample_valid_pipe<= in_valid;
            lut_out_pipe     <= lut_value;
            lut_valid_pipe   <= sample_valid_pipe;
            out_sample       <= lut_out_pipe;
            out_valid        <= lut_valid_pipe;
        end
    end

endmodule

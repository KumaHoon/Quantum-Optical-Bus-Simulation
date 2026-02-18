`timescale 1ns/1ps

module tb_feedforward_lut;
    localparam int DATA_WIDTH = 16;
    localparam int VECTOR_COUNT = 128;
    localparam int PIPE_LATENCY = 2;
    localparam string STIMULUS_FILE = "vectors/stimulus.mem";
    localparam string EXPECTED_FILE = "vectors/expected.mem";

    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic rst_n = 1'b0;
    logic in_valid = 1'b0;
    logic signed [DATA_WIDTH-1:0] in_sample = '0;
    logic out_valid;
    logic signed [DATA_WIDTH-1:0] out_sample;

    logic signed [DATA_WIDTH-1:0] stimulus[0:VECTOR_COUNT-1];
    logic signed [DATA_WIDTH-1:0] expected[0:VECTOR_COUNT-1];

    feedforward_lut #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_sample(in_sample),
        .out_valid(out_valid),
        .out_sample(out_sample)
    );

    integer i;
    integer send_idx;
    integer recv_idx;
    integer cycle;
    integer launch_cycle[0:PIPE_LATENCY];
    logic signed [DATA_WIDTH-1:0] expected_pipe[0:PIPE_LATENCY];
    logic expected_valid[0:PIPE_LATENCY];
    integer measured_latency;
    integer errors;
    integer first_output_cycle;

    task automatic fail_if(
        input bit condition,
        input string msg
    );
        if (condition) begin
            $display("[ERROR] %s at cycle=%0d", msg, cycle);
            errors = errors + 1;
        end
    endtask

    initial begin
        $display("Loading golden vectors...");
        $readmemh(STIMULUS_FILE, stimulus);
        $readmemh(EXPECTED_FILE, expected);

        for (i = 0; i <= PIPE_LATENCY; i = i + 1) begin
            launch_cycle[i] = -1;
            expected_pipe[i] = '0;
            expected_valid[i] = 1'b0;
        end
        measured_latency = -1;
        send_idx = 0;
        recv_idx = 0;
        errors = 0;
        cycle = 0;
        first_output_cycle = -1;
        rst_n = 1'b0;

        $dumpfile("waves.vcd");
        $dumpvars(0, tb_feedforward_lut);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        forever begin
            @(posedge clk);
            cycle = cycle + 1;

            if (send_idx < VECTOR_COUNT) begin
                in_sample <= stimulus[send_idx];
                in_valid <= 1'b1;
                expected_pipe[0] <= expected[send_idx];
                expected_valid[0] <= 1'b1;
                launch_cycle[0] <= cycle;
                send_idx <= send_idx + 1;
            end else begin
                in_valid <= 1'b0;
                expected_valid[0] <= 1'b0;
                launch_cycle[0] <= -1;
            end

            expected_pipe[1] <= expected_pipe[0];
            expected_pipe[2] <= expected_pipe[1];
            expected_valid[1] <= expected_valid[0];
            expected_valid[2] <= expected_valid[1];
            launch_cycle[1] <= launch_cycle[0];
            launch_cycle[2] <= launch_cycle[1];

            if (out_valid !== 1'b0) begin
                fail_if(out_sample !== expected_pipe[PIPE_LATENCY], $sformatf(
                    "output mismatch. expected=%0d actual=%0d", expected_pipe[PIPE_LATENCY], out_sample));
            end

            if (out_valid && recv_idx == 0) begin
                first_output_cycle = cycle;
                measured_latency = cycle - launch_cycle[PIPE_LATENCY];
            end

            if (out_valid) begin
                recv_idx <= recv_idx + 1;
            end

            if (send_idx >= VECTOR_COUNT && recv_idx >= VECTOR_COUNT) begin
                if (errors == 0) begin
                    if (measured_latency < 0) begin
                        $display("[ERROR] No valid output observed.");
                    end else begin
                        $display("[OK] All %0d vectors passed.", VECTOR_COUNT);
                        $display("[OK] Measured cycle latency: %0d", measured_latency);
                    end
                end else begin
                    $display("[ERROR] Test failed with %0d mismatches.", errors);
                end
                $finish;
            end
        end
    end
endmodule


// 一些用于计算的cuda函数

// 从主机端复制数据至设备端
void copyInputToDevice(int input_data_size, int output_data_size)
{
    float *device_input_data = nullptr;
    float *device_output_data = nullptr;
    cudaMalloc((void **)&device_input_data, input_data_size);
    cudaMalloc((void **)device_output_data, output_data_size);
}

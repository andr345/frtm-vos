#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <nppcore.h>
#include <nppi.h>
#include <nppdefs.h>

#ifndef TORCH_CHECK
    // Workaround for older PyTorch, e.g version 1.0.1
    #include <ATen/cuda/CUDAGuard.h>
    #define TORCH_CHECK(x, s)
#else
    #include <c10/cuda/CUDAGuard.h>
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_DEVICE(x,y) TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")
#define CHECK_DIM(x, dim) TORCH_CHECK(x.dim() >= 2, #x " has " #dim " dimensions. At least 2 are required.")
#define CHECK_MODE(mode)

NppiSize _spatial_size(torch::Tensor t)
{
    uint64_t h = t.size(-2);
    uint64_t w = t.size(-1);
    TORCH_CHECK(h < (1<<31) && w < (1<<31), "Spatial size of tensor is too large");
    NppiSize size = {(int)w, (int)h};
    return size;
}

int _mode_to_interpolation(std::string& mode)
{
    int interpolation;
    if (mode == "nearest") {
        interpolation = NPPI_INTER_NN;
    } else if (mode == "bilinear") {
        interpolation = NPPI_INTER_LINEAR;
    } else if (mode == "bicubic") {
        interpolation = NPPI_INTER_CUBIC;
    } else {
        printf("Unknown mode '%s', ", mode.c_str());
        TORCH_CHECK(false, "Use one of 'nearest', 'bilinear' or 'bicubic'");
    }
    return interpolation;
}

template<typename T, int nrows, typename op_type>
void _do_warp(op_type op, torch::Tensor src, torch::Tensor dst, torch::Tensor transform, std::string& mode)
{
    CHECK_INPUT(src);
    CHECK_INPUT(dst);
    CHECK_DEVICE(src, dst);

    NppiSize ssize = _spatial_size(src);
    NppiRect sroi = {0, 0, ssize.width, ssize.height};
    auto _src = src.view({-1, ssize.height, ssize.width});
    unsigned int sstep = _src.stride(-2) * src.dtype().itemsize();

    NppiSize dsize = _spatial_size(dst);
    NppiRect droi = {0, 0,  dsize.width, dsize.height};
    auto _dst = dst.view({-1, dsize.height, dsize.width});
    unsigned int dstep = _dst.stride(-2) * dst.dtype().itemsize();

    TORCH_CHECK(transform.size(0) >= nrows, "transform matrix must have 2 or 3 rows");
    TORCH_CHECK(transform.size(1) == 3, "transform matrix must have 3 columns");

    double H[nrows][3];

    auto _transform = transform.contiguous().to(torch::kFloat64);
    auto t_a = _transform.accessor<double,2>();
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 3; x++) {
            H[y][x] = t_a[y][x];
        }
    }

    int interpolation = _mode_to_interpolation(mode);

    auto stream = at::cuda::getCurrentCUDAStream(src.device().index());
    at::cuda::CUDAStreamGuard guard(stream);
    if (nppGetStream() != stream) {
        nppSetStream(stream);
    }

    auto src_a = src.accessor<T,3>();
    auto dst_a = dst.accessor<T,3>();
    for (int i = 0; i < src_a.size(0); i++) {
        op(src_a[i].data(), ssize, sstep, sroi, dst_a[i].data(), dstep, droi, H, interpolation);
    }
}

void warp_affine(torch::Tensor src, torch::Tensor dst, torch::Tensor transform, std::string mode)
{
    auto dtype = dst.dtype();
    if (dtype == torch::kFloat32) {
        _do_warp<float, 2>(nppiWarpAffine_32f_C1R, src, dst, transform, mode);
    } else if (dtype == torch::kUInt8) {
        _do_warp<unsigned char, 2>(nppiWarpAffine_8u_C1R, src, dst, transform, mode);
    }
    else {
        TORCH_CHECK(false, "Unsupported data type. use either float() or byte()");
    }
}

void warp_perspective(torch::Tensor src, torch::Tensor dst, torch::Tensor transform, std::string mode)
{
    auto dtype = dst.dtype();
    if (dtype == torch::kFloat32) {
        _do_warp<float, 3>(nppiWarpPerspective_32f_C1R, src, dst, transform, mode);
    } else if (dtype == torch::kUInt8) {
        _do_warp<unsigned char, 3>(nppiWarpPerspective_8u_C1R, src, dst, transform, mode);
    }
    else {
        TORCH_CHECK(false, "Unsupported data type. use either float() or byte()");
    }
}

template<typename T, typename op_type>
void _do_remap(op_type op, torch::Tensor src, torch::Tensor dst, torch::Tensor map, std::string& mode)
{
    CHECK_INPUT(src);
    CHECK_INPUT(dst);
    CHECK_DEVICE(src, dst);

    NppiSize ssize = _spatial_size(src);
    NppiRect sroi = {0, 0, ssize.width, ssize.height};
    auto _src = src.view({-1, ssize.height, ssize.width});
    unsigned int sstep = _src.stride(-2) * src.dtype().itemsize();
    auto src_a = src.accessor<T,3>();

    NppiSize dsize = _spatial_size(dst);
    NppiRect droi = {0, 0,  dsize.width, dsize.height};
    auto _dst = dst.view({-1, dsize.height, dsize.width});
    unsigned int dstep = _dst.stride(-2) * dst.dtype().itemsize();
    auto dst_a = dst.accessor<T,3>();

    auto map_a = map.accessor<float,3>();
    auto mx_step = map[0].stride(-2) * map.dtype().itemsize();
    auto my_step = map[1].stride(-2) * map.dtype().itemsize();

    int interpolation = _mode_to_interpolation(mode);

    auto stream = at::cuda::getCurrentCUDAStream(src.device().index());
    at::cuda::CUDAStreamGuard guard(stream);
    if (nppGetStream() != stream) {
        nppSetStream(stream);
    }

    for (int i = 0; i < src_a.size(0); i++) {
        op(src_a[i].data(), ssize, sstep, sroi,
           map_a[0].data(), mx_step, map_a[1].data(), my_step,
           dst_a[i].data(), dstep, dsize, interpolation);
    }
}


void remap(torch::Tensor src, torch::Tensor dst, torch::Tensor map, std::string mode)
{
    auto dtype = dst.dtype();
    if (dtype == torch::kFloat32) {
        _do_remap<float>(nppiRemap_32f_C1R, src, dst, map, mode);
    } else if (dtype == torch::kUInt8) {
        _do_remap<unsigned char>(nppiRemap_8u_C1R, src, dst, map, mode);
    }
    else {
        TORCH_CHECK(false, "Unsupported data type. use either float() or byte()");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warp_affine", &warp_affine, "Apply an affine warp to a 2D-tensor shape=(C,H,W) (CUDA)");
  m.def("warp_perspective", &warp_perspective, "Apply a perspective warp to a 2D-tensor shape=(C,H,W) (CUDA)");
  m.def("remap", &remap, "Remap a 2D-tensor shape(C,sH,sW) with a warping map shape=(2,dH,dW) (CUDA)");
}

#include "LazyAtenFunctions.h"

namespace lazy_mode {
at::Tensor LazyNativeFunctions::mm(const at::Tensor & self, const at::Tensor & mat2) {
    /*    
    if (force_eager_fallback(at::aten::mm)) {
        return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP(mm)>::call(
            self,
            mat2
        );
    }

    TORCH_LAZY_FN_COUNTER("lazy::");
    auto common_device = torch::lazy::GetBackendDevice(self, mat2);
    TORCH_INTERNAL_ASSERT(common_device);
    
    torch::lazy::LazyTensorPtr lazy_self = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
    torch::lazy::LazyTensorPtr lazy_mat2 = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(mat2, *common_device);
    auto out_meta = at::meta::mm(self, mat2);
    std::vector<Shape> shapes{Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    auto node = torch::lazy::MakeNode<ir::ops::Mm>(lazy_self->GetIrValue(),
                          lazy_mat2->GetIrValue(),
                                                                                  std::move(shapes));
    auto result = torch::lazy::CreateAtenFromLtcTensor(
            torch::lazy::LazyTensor::Create(std::move(node), *common_device));
    */
    return at::Tensor();
};

at::Tensor LazyNativeFunctions::relu(const at::Tensor & self) {
    /* 
    if (force_eager_fallback(at::aten::relu)) {
        return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP(relu)>::call(
            self
        );
    }

    TORCH_LAZY_FN_COUNTER("lazy::");
    auto common_device = torch::lazy::GetBackendDevice(self);
    TORCH_INTERNAL_ASSERT(common_device);
    
    torch::lazy::LazyTensorPtr lazy_self = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
    
    auto shapes = torch::lazy::compute_shape_relu(self);
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    auto node = torch::lazy::MakeNode<ir::ops::Relu>(lazy_self->GetIrValue(),
                                                                                  std::move(shapes));
    auto result = torch::lazy::CreateAtenFromLtcTensor(
            torch::lazy::LazyTensor::Create(std::move(node), *common_device));
    return result;
    */
    return at::Tensor();
};

at::Tensor LazyNativeFunctions::add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    /* 
    if (force_eager_fallback(at::aten::add)) {
        return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP2(add, Tensor)>::call(
            self,
            other,
            alpha
        );
    }

    TORCH_LAZY_FN_COUNTER("lazy::");
    auto common_device = torch::lazy::GetBackendDevice(self, other);
    TORCH_INTERNAL_ASSERT(common_device);
    
    torch::lazy::LazyTensorPtr lazy_self = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
    torch::lazy::LazyTensorPtr lazy_other = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(other, *common_device);
    auto out_meta = at::meta::add(self, other, alpha);
    std::vector<Shape> shapes{Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    auto node = torch::lazy::MakeNode<ir::ops::AddTensor>(lazy_self->GetIrValue(),
                          lazy_other->GetIrValue(),
                          torch::lazy::LazyGraphExecutor::Get()->GetIrValueForScalarFromCodegen(alpha),
                                                                                  std::move(shapes));
    auto result = torch::lazy::CreateAtenFromLtcTensor(
            torch::lazy::LazyTensor::Create(std::move(node), *common_device));
    return result;
    */
    return at::Tensor();
};
} // namespace lazy_mode

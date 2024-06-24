#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "vector_cuda.h"


Vector::Vector(double hx,double hy,double hz): h_x(hx), h_y(hy), h_z(hz){
    cudaError(cudaMalloc(&d_x, sizeof(double)), "Failed to allocate the memory for d_x");
    cudaError(cudaMalloc(&d_y, sizeof(double)), "Failed to allocate the memory for d_y");
    cudaError(cudaMalloc(&d_z, sizeof(double)), "Failed to allocate the memory for d_z");

    cudaError(cudaMemcpy(d_x, &h_x, sizeof(double), cudaMemcpyHostToDevice), "Failed to copy the memory from h_x");
    cudaError(cudaMemcpy(d_y, &h_y, sizeof(double), cudaMemcpyHostToDevice), "Failed to copy the memory from h_y");
    cudaError(cudaMemcpy(d_z, &h_z, sizeof(double), cudaMemcpyHostToDevice), "Failed to copy the memory from h_z");
}

Vector::~Vector(){
    cudaError(cudaFree(d_x), "Failed to free memory for d_x");
    cudaError(cudaFree(d_y), "Failed to free memory for d_y");
    cudaError(cudaFree(d_z), "Failed to free memory for d_z");
}

void Vector::cpyToHost(){
    cudaError(cudaMemcpy(&h_x, d_x, sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_x");
    cudaError(cudaMemcpy(&h_y, d_y, sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy the memory form d_y");
    cudaError(cudaMemcpy(&h_z, d_z, sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_z");
}

void Vector::print() const {
    std::cout << "vector: (" << h_x << ", " << h_y << ", " << h_z << ")" << std::endl;
}

Vector Vector::add(const Vector &vec){
    Vector result(0.0, 0.0, 0.0);
    addKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch addKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::add(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    addNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch addNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::radd(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    raddNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch raddNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::sub(const Vector &vec){
    Vector result(0.0, 0.0, 0.0);
    subKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch subKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::sub(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    subNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch subNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::rsub(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    rsubNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Faield to launch rsubNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::mul(const Vector &vec){
    Vector result(0.0, 0.0, 0.0);
    mulKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, result.d_x, result.d_y,result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch mulKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::mul(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    mulNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch mulNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::rmul(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    rmulNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch rmulNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::div(const Vector &vec){
    Vector result(0.0, 0.0, 0.0);
    divKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, result.d_x, result.d_y,result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch divKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::div(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    divNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_y, result.d_y, result.d_z);
    cudaError(cudaGetLastError(),"Failed to launch divNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::rdiv(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    rdivNumKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch rdivNumKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::scale(const double &scalar){
    Vector result(0.0, 0.0, 0.0);
    scaleKernel<<<1,1>>>(d_x, d_y, d_z, scalar, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch scaleKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

Vector Vector::cross(const Vector &vec){
    Vector result(0.0, 0.0, 0.0);
    crossKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch crossKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

double Vector::dot(const Vector &vec){
    double h_result;
    double *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(double)), "Failed to allocate the memory for d_result");
    dotKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch dotKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

double Vector::norm(){
    double h_result;
    double *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(double)), "Failed to allocate the memory for d_result");
    normKernel<<<1,1>>>(d_x, d_y, d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch normKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

Vector Vector::proj(const Vector &onto){
    Vector result(0.0, 0.0, 0.0);
    projKernel<<<1,1>>>(d_x, d_y, d_z, onto.d_x, onto.d_y, onto.d_z, result.d_x, result.d_y, result.d_z);
    cudaError(cudaGetLastError(), "Failed to launch projKernel");
    cudaDeviceSynchronize();
    result.cpyToHost();
    return result;
}

bool Vector::operator==(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    eqKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch eqKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::operator!=(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    neKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch neKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::operator>=(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    geKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch geKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::operator>(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    gtKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch gtKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::operator<=(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    leKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to free memory for d_result");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Faild to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::operator<(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_result");
    ltKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to free memory for d_result");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to free memory for d_result");
    cudaError(cudaFree(d_result), "Failed to free memory for d_result");
    return h_result;
}

bool Vector::eq(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    eqKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch eqKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

bool Vector::ne(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    neKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch neKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

bool Vector::gt(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    gtKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch gtKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

bool Vector::ge(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    geKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch geKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

bool Vector::lt(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    ltKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch ltKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

bool Vector::le(const Vector &vec) const{
    bool h_result;
    bool *d_result;
    cudaError(cudaMalloc(&d_result, sizeof(bool)), "Failed to allocate the memory for d_reuslt");
    leKernel<<<1,1>>>(d_x, d_y, d_z, vec.d_x, vec.d_y, vec.d_z, d_result);
    cudaError(cudaGetLastError(), "Failed to launch leKernel");
    cudaDeviceSynchronize();
    cudaError(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost), "Failed to copy the memory from d_result");
    cudaError(cudaFree(d_result), "Failed to free the memory for d_result");
    return h_result;
}

Vector Vector::zeros(){
    return Vector(0.0, 0.0, 0.0);
}

Vector Vector::ones(){
    return Vector(1.0, 1.0, 1.0);
}

Vector Vector::fill(const double &value){
    return Vector(value, value, value);
}

__global__ void addKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 + *d_x2;
    *r_d_y = *d_y1 + *d_y2;
    *r_d_z = *d_z1 + *d_z2;
}

__global__ void addNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 + scalar;
    *r_d_y = *d_y1 + scalar;
    *r_d_z = *d_z1 + scalar;
}

__global__ void raddNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = scalar + *d_x1;
    *r_d_y = scalar + *d_y1;
    *r_d_z = scalar + *d_z1;
}

__global__ void subKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 - *d_x2;
    *r_d_y = *d_y1 - *d_y2;
    *r_d_z = *d_z1 - *d_z2;
}


__global__ void subNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 - scalar;
    *r_d_y = *d_y1 - scalar;
    *r_d_z = *d_z1 - scalar;
}

__global__ void rsubNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = scalar - *d_x1;
    *r_d_y = scalar - *d_y1;
    *r_d_z = scalar - *d_z1;
}

__global__ void mulKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 * *d_x2;
    *r_d_y = *d_y1 * *d_y2;
    *r_d_z = *d_z1 * *d_z2;
}

__global__ void mulNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 * scalar;
    *r_d_y = *d_y1 * scalar;
    *r_d_z = *d_z1 * scalar;
}

__global__ void rmulNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = scalar * *d_x1;
    *r_d_y = scalar * *d_y1;
    *r_d_z = scalar * *d_z1;
}

__global__ void divKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 / *d_x2;
    *r_d_y = *d_y1 / *d_y2;
    *r_d_z = *d_z1 / *d_z2;
}

__global__ void divNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 / scalar;
    *r_d_y = *d_y1 / scalar;
    *r_d_z = *d_z1 / scalar;
}

__global__ void rdivNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = scalar / *d_x1;
    *r_d_y = scalar / *d_y1;
    *r_d_z = scalar / *d_z1;
}

__global__ void scaleKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = *d_x1 * scalar;
    *r_d_y = *d_y1 * scalar;
    *r_d_z = *d_z1 * scalar;
}

__global__ void crossKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    *r_d_x = (*d_y1 * *d_z2) - (*d_z1 * *d_y2);
    *r_d_y = (*d_z1 * *d_x2) - (*d_x1 * *d_z2);
    *r_d_z = (*d_x1 * *d_y2) - (*d_y1 * *d_x2);
}

__global__ void dotKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *result){
    *result = (*d_x1 * *d_x2) + (*d_y1 * *d_y2) + (*d_z1 * *d_z2);
}

__global__ void normKernel(double *d_x, double *d_y, double *d_z, double *result){
    *result = sqrt((*d_x * *d_x) + (*d_y * *d_y) + (*d_z * *d_z));
}

__global__ void eqKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isEq){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    const double epsilon = 1e-8;
    *isEq = (fabs(norm1 - norm2) < epsilon);
}

__global__ void neKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isNe){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    const double epsilon = 1e-8;
    *isNe = (fabs(norm1 - norm2) >= epsilon);
}

__global__ void gtKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isGt){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    *isGt = (norm1 > norm2);
}

__global__ void geKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isGe){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    *isGe = (norm1 >= norm2);
}

__global__ void ltKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isLt){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    *isLt = (norm1 < norm2);
}

__global__ void leKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isLe){
    double norm1 = sqrt((*d_x1 * *d_x1) + (*d_y1 * *d_y1) + (*d_z1 * *d_z1));
    double norm2 = sqrt((*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z2 * *d_z2));
    *isLe = (norm1 <= norm2);
}

__global__ void unitKernel(double *d_x, double *d_y, double *d_z, double *r_d_x, double *r_d_y, double *r_d_z){
    double norm = sqrt((*d_x * *d_x) + (*d_y * *d_y) + (*d_z * *d_z));
    if(norm != 0.0){
        *r_d_x = *d_x / norm;
        *r_d_y = *d_y / norm;
        *r_d_z = *d_z / norm;
    }else{
        *r_d_x = 0.0;
        *r_d_y = 0.0;
        *r_d_z = 0.0;
        printf("Error: Attempted to normalize a zero vector!\n");
    }
}

__global__ void projKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z){
    double dotProd1 = (*d_x1 * *d_x2) + (*d_y1 * *d_y2) + (*d_z1 * *d_z2);
    double dotProd2 = (*d_x2 * *d_x2) + (*d_y2 * *d_y2) + (*d_z1 * *d_z2);
    double scalar = dotProd1 / dotProd2;
    *r_d_x = scalar * *d_x2;
    *r_d_y = scalar * *d_y2;
    *r_d_z = scalar * *d_z2;
}


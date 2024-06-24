#ifndef VECTOR_H
#define VECTOR_H

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cuda_runtime.h>
#include <iterator>


class Vector{
    public:
        double h_x,h_y,h_z;
        double *d_x,*d_y,*d_z;

        void cudaError(cudaError_t err,const char *msg) const{
            if(err != cudaSuccess){
                std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    public:
        Vector(double hx,double hy,double hz);
        ~Vector();

        void cpyToHost();
        void print() const;
        std::string str() const;

        Vector add(const Vector &vec);
        Vector add(const double &scalar);
        Vector radd(const double &scalar);

        Vector sub(const Vector &vec);
        Vector sub(const double &scalar);
        Vector rsub(const double &scalar);

        Vector mul(const Vector &vec);
        Vector mul(const double &scalar);
        Vector rmul(const double &scalar);

        Vector div(const Vector &vec);
        Vector div(const double &scalar);
        Vector rdiv(const double &scalar);
        
        Vector scale(const double &scalar);
        Vector cross(const Vector &vec);
        double dot(const Vector &vec);
        double norm();
        double unit();
        Vector proj(const Vector &onto);

        bool operator==(const Vector &vec) const;
        bool operator!=(const Vector &vec) const;
        bool operator>=(const Vector &vec) const;
        bool operator>(const Vector &vec) const;
        bool operator<=(const Vector &vec) const;
        bool operator<(const Vector &vec) const;

        bool eq(const Vector &vec) const;
        bool ne(const Vector &vec) const;
        bool gt(const Vector &vec) const;
        bool ge(const Vector &vec) const;
        bool lt(const Vector &vec) const;
        bool le(const Vector &vec) const;

        static Vector zeros();
        static Vector ones();
        static Vector fill(const double &value);
};



__global__ void addKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void addNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void raddNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);

__global__ void subKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void subNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void rsubNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);

__global__ void mulKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void mulNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void rmulNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar,  double *r_d_x, double *r_d_y, double *r_d_z);

__global__ void divKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void divNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void rdivNumKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);

__global__ void scaleKernel(double *d_x1, double *d_y1, double *d_z1, double scalar, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void dotKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *z_x2, double *d_z2, double *result);
__global__ void crossKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void normKernel(double *d_x, double *d_y, double *d_z, double *result);
__global__ void unitKernel(double *d_x, double *d_y, double *d_z, double *r_d_x, double *r_d_y, double *r_d_z);
__global__ void projKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, double *r_d_x, double *r_d_y, double *r_d_z);

__global__ void eqKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isEq);
__global__ void neKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isNe);
__global__ void gtKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isGt);
__global__ void geKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isGe);
__global__ void ltKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isLt);
__global__ void leKernel(double *d_x1, double *d_y1, double *d_z1, double *d_x2, double *d_y2, double *d_z2, bool *isLe);

#endif


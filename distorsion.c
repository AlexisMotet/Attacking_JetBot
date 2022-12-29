#include <stddef.h>
#include <stdio.h>
#include <math.h>

// gcc -fPIC -shared -Wall -o distorsion.so distorsion.c

typedef struct cam_mtx
{
    float fx;
    float fy;
    float cx;
    float cy;
} cam_mtx;

typedef struct dist_coefs
{
    float k1;
    float k2;
    float k3;
} dist_coefs;

void cdistort(const float * image, const size_t row, const size_t col, const cam_mtx * mtx,
          const dist_coefs * coefs, unsigned *map, float * image_distorded) 
{
    size_t x, y, c;
    float nx, ny, r2, ndx_f, ndy_f;
    unsigned dx, dy;
    int i, i_v;
    for (x = 0; x < col; ++x)
    {
        for (y = 0; y < row; ++y)
        {
            nx = (x - mtx->cx)/mtx->fx;
            ny = (y - mtx->cy)/mtx->fy;

            r2 = pow(nx, 2) + pow(ny, 2);
            ndx_f = nx + nx * (coefs->k1*r2 + coefs->k2*pow(r2, 2) + coefs->k3*pow(r2, 3));
            ndy_f = ny + ny * (coefs->k1*r2 + coefs->k2*pow(r2, 2) + coefs->k3*pow(r2, 3));
            
            dx = ndx_f * mtx->fx + mtx->cx;
            dy = ndy_f * mtx->fy + mtx->cy;
            if (0 <= dx && dx < col && 0 <= dy && dy < row)
            {
                i_v = x + y * col;
                i = dx + dy * col;    
                for (c = 0; c < 3; c++)
                {   
                    image_distorded[i + c * row * col] = image[i_v + c * row * col];
                }
                map[i] = i_v;
            }
        }
    }
}

void cdistort_with_map(const float * image, const size_t row, const size_t col,
                       const unsigned *map, float * image_distorded)
{
    int x, y, c;
    int i, i_v;
    for (x = 0; x < col; ++x)
    {
        for (y = 0; y < row; ++y)
        {
            i = x + y * col;
            i_v = map[x + y * col];
            for (c = 0; c < 3; c++)
            {
                image_distorded[i + c * row * col] = image[i_v + c * row * col];
            }
        }
    }
}

void cundistort(const float * image_distorded, const size_t row, const size_t col,
                const unsigned *map, float * image)
{
    int x, y, c;
    int i, i_v;
    for (x = 0; x < col; ++x)
    {
        for (y = 0; y < row; ++y)
        {
            i = x + y * col;
            i_v = map[x + y * col];
            for (c = 0; c < 3; c++)
            {
                image[i_v + c * row * col] = image_distorded[i + c * row * col];
            }
        }
    }
}

#pragma once
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;

// A simple statically sized matrix class
template<size_t h, size_t w, class T>
class Mat {
  private:

  // everything is public because im "responsible"
  public:
    std::array<T, h*w> data;

    // initialize from list
    Mat(initializer_list<T> aa) : data() {
      array<T, h*w> a;
      copy_n(aa.begin(), h*w, a.begin());
      for(size_t x = 0; x < w; x++) {
        for(size_t y = 0; y < h; y++) {
          (*this)(x, y) = a[x * w + y];
        }
      }
    }

    // in row major form
    Mat(std::array<T, h*w> a) : data() {
      for(size_t x = 0; x < w; x++) {
        for(size_t y = 0; y < h; y++) {
          (*this)(x, y) = a[x * w + y];
        }
      }
    }

    Mat(Mat<h, w, T>&& left) {
      move(left.data.begin(), left.data.end(), data.begin());
    }

    T* d() {
      return data.data();
    }

    const T* d() const {
      return data.data();
    }

    // TODO: equals operator for initializer_list
    Mat(const Mat<h, w, T>& left) = default;
    Mat<h, w, T>& operator=(const Mat<h, w, T>& left) = default;

    Mat() : data() {};

    T& operator()(size_t x, size_t y) {
      return data[y * w + x];
    }

    const T& operator() (size_t x, size_t y) const {
      return data[y * w + x];
    }

    template<size_t ww = w,
      class = typename std::enable_if<ww == 1>::type >
    T& operator()(size_t x) {
      return data[x];
    }

    template<size_t ww = w,
      class = typename std::enable_if<ww == 1>::type >
    const T& operator()(size_t x) const {
      return data[x];
    }

    template<size_t hh, size_t ww,
      class = typename std::enable_if<ww <= w && hh <= h>::type >
    Mat<hh, ww, T> resize() {
      Mat<hh, ww, T> r;
      for(size_t x = 0; x < ww; x++) {
        for(size_t y = 0; y < hh; y++) {
          r(x, y) = (*this)(x, y);
        }
      }
      return r;
    }

    template<class TT, class F>
    Mat<h, w, TT> element_wise(const Mat<h, w, T>& right, F op) const {
      Mat<h, w, TT> r;
      for(size_t i = 0; i < h; i++) {
        for(size_t j = 0; j < w; j++) {
          r(i, j) = op((*this)(i, j), right(i, j));
        }
      }
      return r;
    }

    template<class TT, class F>
    Mat<h, w, TT>&& element_wise(Mat<h, w, T>&& right, F op) const {
      for(size_t i = 0; i < h; i++) {
        for(size_t j = 0; j < w; j++) {
          right(i, j) = op((*this)(i, j), right(i, j));
        }
      }
      return move(right);
    }

    Mat<h, w, T> operator+=(const Mat<h, w, T>& right) {
      *this = *this + right;
      return *this;
    }

    Mat<h, w, T> emul(const Mat<h, w, T>& right) const {
      return element_wise<T>(right, [](T a, T b) {return a * b;});
    }

    Mat<h, w, T>&& emul(Mat<h, w, T>&& right) const {
      return element_wise<T>(move(right), [](T a, T b) {return a * b;});
    }

    template<class TT, class F>
    Mat<h, w, TT> element_each(F op) const & {
      Mat<h, w, TT> r;
      for(size_t i = 0; i < h; i++) {
        for(size_t j = 0; j < w; j++) {
          r(i, j) = op((*this)(i, j));
        }
      }
      return r;
    }

    template<class TT, class F>
    Mat<h, w, TT>&& element_each(F op) && {
      for(size_t i = 0; i < h; i++) {
        for(size_t j = 0; j < w; j++) {
          (*this)(i, j) = op((*this)(i, j));
        }
      }
      return move(*this);
    }

    Mat<h, w, T> operator*(T right) const & {
      return element_each<T>([right](T a) -> T {return a * right;});
    }

    Mat<h, w, T>&& operator*(T right) && {
      return move(*this).template element_each<T>([right](T a) -> T {return a * right;});
    }

    Mat<h, w, T> operator/(T right) const & {
      return element_each<T>([right](T a) -> T {return a / right;});
    }

    Mat<h, w, T>&& operator/(T right) && {
      return move(*this).template element_each<T>([right](T a) -> T {return a / right;});
    }

    template<size_t ww = w,
      class = typename enable_if<ww == 1>::type >
    T magnitude() const {
      Mat<h, w, T> m = (*this).emul(*this);
      T init = 0;
      T a = accumulate(m.data.begin(), m.data.end(), init);
      return sqrt(a);
    }

    template<size_t ww = w,
      class = typename enable_if<ww == 1>::type >
    Mat<h, w, T> normalize() const & {
      Mat<h, w, T> m;
      T l = magnitude();
      transform(data.begin(), data.end(), m.data.begin(), [l](const T& x){return x/l;});
      return m;
    }

    template<size_t ww = w,
      class = typename enable_if<ww == 1>::type >
    Mat<h, w, T>&&  normalize() && {
      T l = magnitude();
      transform(data.begin(), data.end(), data.begin(), [l](const T& x){return x/l;});
      return move(*this);
    }

    Mat<w, h, T> transpose() {
      Mat<w, h, T> res;
      for(size_t x = 0; x < w; x++) {
        for(size_t y = 0; y < h; y++) {
          res(x, y) = operator()(y, x);
        }
      }
      return res;
    }
};

// global binary ops

template<size_t h, size_t ww, size_t w, class T>
Mat<h, ww, T>&& operator*(Mat<w, ww, T>&& left, const Mat<w, ww, T>& right) {
  for(size_t i = 0; i < h; i++) {
    for(size_t j = 0; j < ww; j++) {
      T s = 0;
      for(size_t k = 0; k < w; k++) {
        s += left(i, k) * right(k, j);
      }
      left(i, j) = s;
    }
  }
  return move(left);
}

// PLUS
template<size_t h, size_t w, class T>
Mat<h, w, T> operator+(const Mat<h, w, T>& left, const Mat<h, w, T>& right) {
  return left.template element_wise<T>(right, plus<T>());
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator+(const Mat<h, w, T>& left, Mat<h, w, T>&& right) {
  return left.template element_wise<T>(move(right), plus<T>());
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator+(Mat<h, w, T>&& left, const Mat<h, w, T>& right) {
  return right.template element_wise<T>(move(left), plus<T>());
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator+(Mat<h, w, T>&& left, Mat<h, w, T>&& right) {
  return left.template element_wise<T>(move(right), plus<T>());
}

//MINUS
template<size_t h, size_t w, class T>
Mat<h, w, T> operator-(const Mat<h, w, T>& left, const Mat<h, w, T>& right) {
  return left.template element_wise<T>(right, [](T a, T b) {return a - b;});
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator-(const Mat<h, w, T>& left, Mat<h, w, T>&& right) {
  return left.template element_wise<T>(move(right), [](T a, T b) {return a - b;});
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator-(Mat<h, w, T>&& left, const Mat<h, w, T>& right) {
  return right.template element_wise<T>(move(left), [](T a, T b) {return b - a;});
}

template<size_t h, size_t w, class T>
Mat<h, w, T>&& operator-(Mat<h, w, T>&& left, Mat<h, w, T>&& right) {
  return left.template element_wise<T>(move(right), [](T a, T b) {return a - b;});
}

//MULT
template<size_t h, size_t w, size_t ww, class T>
Mat<h, ww, T> operator*(const Mat<h, w, T>& left, const Mat<w, ww, T>& right) {
  Mat<h, ww, T> r;
  for(size_t i = 0; i < h; i++) {
    for(size_t j = 0; j < ww; j++) {
      T s = 0;
      for(size_t k = 0; k < w; k++) {
        s += left(i, k) * right(k, j);
      }
      r(i, j) = s;
    }
  }
  return r;
}


template<size_t h, size_t w, class T>
ostream& operator<<(ostream& os, Mat<h, w, T>& mat) {
  for(size_t x = 0; x < h; x++) {
    for(size_t y = 0; y < w; y++) {
      os << mat(x, y) << ' ';
    }
    os << std::endl;
  }
  return os;
}

template<size_t h, size_t w, class T>
const ostream& operator<<(const ostream& os, Mat<h, w, T> mat) {
  os << mat;
  return os;
}

template<class T>
Mat<3, 1, T> cross(Mat<3, 1, T> left, Mat<3, 1, T> right) {
  Mat<3, 1, T> r;
  T a = left(0);
  T b = left(1);
  T c = left(2);
  T d = right(0);
  T e = right(1);
  T f = right(2);
  r(0) = b*f - c*e;
  r(1) = c*d - a*f;
  r(2) = a*e - b*d;
  return r;
}

template<size_t h, class T>
T dot(const Mat<h, 1, T>& left, const Mat<h, 1, T>& right) {
  Mat<h, 1, T> r = left.emul(right);
  T init = 0;
  return accumulate(r.data.begin(), r.data.end(), init);
}

template<size_t h, class T>
T dot(Mat<h, 1, T>&& left, const Mat<h, 1, T>& right) {
  Mat<h, 1, T> r = right.emul(left);
  T init = 0;
  return accumulate(r.data.begin(), r.data.end(), init);
}

template<class T>
Mat<4, 1, T> cross(Mat<4, 1, T> left, Mat<4, 1, T> right) {
  Mat<3, 1, T> l = left.resize();
  Mat<3, 1, T> r = right.resize();
  Mat<3, 1, T> cr = cross(l, r);
  Mat<4, 1, T> r2;
  r2(0) = cr(0);
  r2(1) = cr(1);
  r2(2) = cr(2);
  r2(3) = 1;
  return r2;
}

template<size_t h, class T>
float angle(Mat<h, 1, T> a, Mat<h, 1, T> b) {
  float d = dot(a, b);
  return acos(d/(a.magnitude()*b.magnitude()));
}

template<size_t h, size_t w, class T>
Mat<h, w, T> power(Mat<h, w, T> m, size_t p) {
  return m.template element_each<T>([p](T a) -> T {return pow(a, p);});
}

template<class T>
Mat<4, 4, T> rotate(float deg, float x, float y, float z) {
  float c = cos(-deg * M_PI/180);
  float s = sin(-deg * M_PI/180);
  float t = 1 - cos(-deg * M_PI/180);
  return Mat<4, 4, T>({
      t*x*x + c,    t*x*y + s*z,  t*x*z - s*y,  0,
      t*x*y - s*z,  t*y*y + c,    t*y*z + s*x,  0,
      t*x*z + s*y,  t*y*z - s*x,  t*z*z + c,    0,
      0,            0,            0,            1
    });
}

template<class T>
Mat<3, 3, T> rotate3(float deg, float x, float y, float z) {
  float c = cos(-deg * M_PI/180);
  float s = sin(-deg * M_PI/180);
  float t = 1 - cos(-deg * M_PI/180);
  return Mat<3, 3, T>({
      t*x*x + c,    t*x*y + s*z,  t*x*z - s*y,
      t*x*y - s*z,  t*y*y + c,    t*y*z + s*x,
      t*x*z + s*y,  t*y*z - s*x,  t*z*z + c
    });
}

template<class T>
Mat<4, 4, T> scale(T x, T y, T z) {
  return Mat<4, 4, T>({
      x, 0, 0, 0,
      0, y, 0, 0,
      0, 0, z, 0,
      0, 0, 0, 1
    });
}

template<class T>
Mat<4, 4, T> translate(T x, T y, T z) {
  return Mat<4, 4, T>({
      1, 0, 0, x,
      0, 1, 0, y,
      0, 0, 1, z,
      0, 0, 0, 1
    });
}

template<class T>
Mat<4, 4, T> lookAt(T eyeX, T eyeY, T eyeZ, T centerX, T centerY, T centerZ, T upX, T upY, T upZ) {
  Mat<3, 1, T> g({
    eyeX - centerX,
    eyeY - centerY,
    eyeZ - centerZ
  });
  Mat<3, 1, T> w = g.normalize();
  Mat<3, 1, T> t({upX, upY, upZ});
  Mat<3, 1, T> u = cross(t, w).normalize();
  Mat<3, 1, T> v = cross(w, u);

  Mat<4, 4, T> out = Mat<4, 4, T>({
        u(0), u(1), u(2), 0,
        v(0), v(1), v(2), 0,
        w(0), w(1), w(2), 0,
        0, 0, 0, 1
      }) * translate(-eyeX, -eyeY, -eyeZ);

  return out;
}

template<class T>
Mat<4, 4, T> ortho(T l, T r, T t, T b, T n, T f) {
  return Mat<4, 4, T>({
      2/(r-l),  0,        0,        -2*l/(r-l)-1,
      0,        2/(t-b),  0,        -2*b/(t-b)-1,
      0,        0,        2/(n-f),  2*f/(n-f)+1,
      0,        0,        0,        1
    });
}

template<class T>
Mat<4, 4, T> perspective(T fov, T aspect, T n, T f) {
  T ymax = n * tan(fov * M_PI / 360.0);
  T xmax = ymax * aspect;
  return ortho(-xmax, xmax, -ymax, ymax, n, f) * Mat<4, 4, T>({
      n,  0,  0,    0,
      0,  -n,  0,    0,
      0,  0,  n+f,  f*n,
      0,  0,  -1,    0
    });
}

template<class T>
Mat<3, 1, T> divide(Mat<4, 1, T> m) {
  return Mat<3, 1, T>({
      m(0)/m(3),
      m(1)/m(3),
      m(2)/m(3)
    });
}


// some typedefs
template<size_t h, typename T>
using Vec = Mat<h, 1, T>;

template<typename T>
using Vec2 = Mat<2, 1, T>;
typedef Vec2<float> Vec2f;

template<typename T>
using Vec3 = Mat<3, 1, T>;
typedef Vec3<float> Vec3f;
typedef Vec3<unsigned char> Vec3b;

template<typename T>
using Vec4 = Mat<4, 1, T>;
typedef Vec4<float> Vec4f;

template<typename T>
using Mat2 = Mat<2, 2, T>;
typedef Mat2<float> Mat2f;

template<typename T>
using Mat3 = Mat<3, 3, T>;
typedef Mat3<float> Mat3f;

template<typename T>
using Mat4 = Mat<4, 4, T>;
typedef Mat4<float> Mat4f;

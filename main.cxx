#include "matplotlibcpp.h"
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <iostream>

namespace plt = matplotlibcpp;

using Point = Eigen::Vector2d;

constexpr std::size_t total = 100;
const Point MIN = {0, 0};
const Point MAX = {800, 600};
const auto DIMENSIONS = MIN.rows();
constexpr double PI = 3.14159265359;

constexpr std::array colours = { "r", "g", "b", "c", "m", "y"};

std::vector<Eigen::Vector2d> stars(const std::size_t& clusters) {
  std::vector<Eigen::Vector2d> ret;
  for (auto i = 0u; i < clusters; i++) {
    const auto angle = 2 * PI * i / clusters;
    const auto r = std::min(MAX[0], MAX[1]) / 2;
    ret.emplace_back(
        r * std::cos(angle) + MAX[0] / 2,
        r * std::sin(angle) + MAX[1] / 2
    );
  }
  return ret;
}

std::vector<Point> generate() {
  std::vector<Point> ret;
  std::random_device rd;
  std::mt19937 gen(rd());

  constexpr auto N = 4u;

  const auto PART = (MAX - MIN) / N;
  std::array<std::vector<std::normal_distribution<double>>, N> dis;
  const auto p = stars(N);
  for (auto i = 0u; i < N; i++) {
    for (auto j = 0u; j < DIMENSIONS; j++) {
      dis[i].emplace_back(
          p[i][j],
          PART[j] / 3.
      );
    }
  }

  for (auto i = 0u; i < total; i++) {
    Point point;
    for (auto j = 0u; j < DIMENSIONS; j++) {
      point[j] = dis[i % N][j](gen);
    }
    ret.push_back(std::move(point));
  }
  return ret;
}

std::array<std::vector<double>, 2> divide(
    const std::vector<Eigen::Vector2d>& vec
) {
  std::array<std::vector<double>, 2> ret;
  for (const auto& pt : vec) {
    for (auto i = 0u; i < 2u; i++) {
      ret[i].emplace_back(pt[i]);
    }
  }
  return ret;
}

double distance(const Point& p1, const Point& p2) {
  const auto pDiff = p1 - p2;
  double sum = 0.;
  for (auto i = 0u; i < DIMENSIONS; i++) {
    sum += pDiff[i] * pDiff[i];
  }
  return sum;
}

template <typename Points, typename Centroids>
std::vector<std::vector<Eigen::Vector2d>> distribute(
    const Points& points,
    const Centroids& centroids
) {
  std::vector<std::vector<Eigen::Vector2d>> groups(centroids.size());
  for (const auto& pt : points) {
    std::tuple<double, std::size_t> min {
        std::numeric_limits<double>::max(),
        std::numeric_limits<std::size_t>::quiet_NaN(),
    };
    for (auto j = 0u; j < centroids.size(); j++) {
      const auto dis = distance(
          centroids[j],
          pt
      );
      if (dis < std::get<0>(min)) {
        min = std::make_tuple(
            dis,
            j
        );
      }
    }
    groups[std::get<1>(min)].emplace_back(pt);
  }
  return groups;
}

template <typename Points, typename Centroid>
double sumInOneGroup(const Points& points, const Centroid& centroid) {
  double sum = 0.;
  for (const auto& pt : points) {
    sum += distance(
        centroid,
        pt
    );
  }
  return sum;
}

bool isnan(const Point& point) {
  for (auto k = 0u; k < DIMENSIONS; k++) {
    if (std::isnan(point[k])) {
      return true;
      break;
    }
  }
  return false;
}

template <typename Groups, typename Centroids>
double sumInAllGroups(const Groups& groups, const Centroids& centroids) {
  double sum = 0.;
  for (auto j = 0u; j < groups.size(); j++) {
    if (isnan(centroids[j])) {
      continue;
    }
    sum += sumInOneGroup(groups[j], centroids[j]);
  }
  return sum;
}

template <typename Groups>
std::vector<Point> medians(const Groups& groups) {
  std::vector<Point> meds;
  for (const auto& group : groups) {
    Eigen::Vector2d avg = { 0, 0 };
    for (const auto& pt : group) {
      avg += pt;
    }
    if (group.size() == 0) {
      meds.emplace_back(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
      );
    } else {
      meds.emplace_back(avg / group.size());
    }
  }
  return meds;
}

int main() {

  const auto points = generate();

  std::size_t clusters = 1;
  std::vector<double> values;
  for (auto i = clusters; i < colours.size(); i++) {
    const auto centroids = stars(i);
    const auto groups = distribute(points, centroids);
    const auto meds = medians(groups);
    auto sum = sumInAllGroups(groups, meds);
    values.push_back(std::move(sum));
  }
  double min = std::numeric_limits<double>::max();
  for (auto i = 1u; i < values.size() - 1; i++) {
    const auto d =
        std::abs((values[i + 1] - values[i]) / (values[i] - values[i - 1]));
    if (d < min) {
      min = d;
      clusters = i + 1;
    }
  }

  auto centroids = stars(clusters);
  decltype(centroids) prevCentroids;

  while (prevCentroids != centroids) {
    const auto groups = distribute(points, centroids);
    for (auto i = 0u; i < groups.size(); i++) {
      const auto [x, y] = divide(groups[i]);
      if (groups.size() > colours.size()) {
        plt::plot(x, y, "ko");
      } else {
        plt::plot(x, y, std::string(colours[i]) + std::string("o"));
      }
    }
    const auto [cX, cY] = divide(centroids);
    for (auto i = 0u; i < centroids.size(); i++) {
      decltype(cX) x = {cX[i]};
      decltype(cY) y = {cY[i]};
      plt::plot(
          x,
          y,
          colours[i] + std::string("*"),
          {{"markersize", "16"}}
      );
    }
    plt::show();
    prevCentroids = centroids;
    {
      const auto newCentroids = medians(groups);
      for (auto i = 0u; i < groups.size(); i++) {
        if (isnan(newCentroids[i])) {
          continue;
        }
        centroids[i] = newCentroids[i];
      }
    }
  }
}

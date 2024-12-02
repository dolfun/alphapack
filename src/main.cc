#include <fmt/core.h>
#include <fmt/ranges.h>
#include <cpr/cpr.h>

int main() {
  std::vector<float> v = { 0.25f, 1.25f, 0.55f, 0.43f, 0.69f };
  cpr::Buffer buf = { 
    reinterpret_cast<char*>(v.data()),
    reinterpret_cast<char*>(v.data()) + v.size() * sizeof(v[0]),
    "data.bin"
  };

  auto response = cpr::Post(
    cpr::Url { "0.0.0.0:8000" },
    cpr::Multipart {
      { "type", "inference" },
      { "data", std::move(buf) }
    }
  );

  fmt::println("URL: {}", std::string(response.url));
  fmt::println("Status Code: {}", response.status_code);
  fmt::println("Content Type: {}", response.header["content-type"]);
  fmt::println("Time elapsed: {:.2} ms", response.elapsed * 1000.0f);

  std::vector<float> a(response.text.size() / sizeof(float));
  std::memcpy(a.data(), response.text.data(), response.text.size());
  fmt::println("Response: {}", a);
}
#pragma once
#include "hh/core/id128.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hh {

// set once per process (cluster-wide secrets)
void set_secret_keys(std::array<std::uint8_t,32> K_id,
                     std::array<std::uint8_t,32> K_fp,
                     std::array<std::uint8_t,32> K_idx);

// normalized bytes -> 128-bit ID (keyed)
Id128 id128_for(std::string_view norm);

// derive a 16-bit lobby fingerprint from ID (CHK)
std::uint16_t fp16_from_id(const Id128& id);

// indices from ID; B must be power-of-two if you bit-mask
struct IdxPair { std::size_t i1, i2; };
IdxPair indices_from_id(const Id128& id, std::uint16_t fp16, std::size_t B);

} // namespace hh


#include "base/request.h"

namespace Ramulator {

std::map<std::string, AiMISR> AiMISRInfo::opcode_str_to_aim_ISR;
std::map<Opcode, std::string> AiMISRInfo::aim_opcode_to_str;

std::map<std::string, Type> AiMISRInfo::str_to_type;
std::map<Type, std::string> AiMISRInfo::type_to_str;

std::map<std::string, MemAccessRegion> AiMISRInfo::str_to_mem_access_region;
std::map<MemAccessRegion, std::string> AiMISRInfo::mem_access_region_to_str;

std::string trace_file_path;
bool use_trace_file_path;

} // namespace Ramulator

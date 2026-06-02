// gguf-extract-meta: extract metadata-only GGUF from a full GGUF model file
// or download just the GGUF header from a URL via HTTP(S) Range requests.
//
// Usage:
//   gguf-extract-meta <input.gguf> [output.meta.gguf]
//   gguf-extract-meta <https://.../model.gguf> [output.meta.gguf]

#include "gguf.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "http.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HTTP(S) Range download via common http client (cpp-httplib)
// ---------------------------------------------------------------------------

static bool download_header(const std::string & url,
                            std::vector<uint8_t> & out_data) {
    const int64_t MIN_CHUNK = 64 * 1024;
    const int64_t MAX_CHUNK = 32 * 1024 * 1024;

    try {
        auto [cli, parts] = common_http_client(url);
        cli.set_read_timeout(60, 0);
        cli.set_write_timeout(60, 0);
        cli.set_connection_timeout(30, 0);

        for (int64_t chunk = MIN_CHUNK; chunk <= MAX_CHUNK; chunk *= 2) {
            fprintf(stderr, "Downloading header: bytes 0-%lld (%.0f KiB)... ",
                    (long long)(chunk - 1), chunk / 1024.0);

            httplib::Headers headers = {
                {"Range", "bytes=0-" + std::to_string(chunk - 1)}
            };

            std::vector<uint8_t> body;
            auto res = cli.Get(parts.path, headers,
                [&](const char * data, size_t len) {
                    body.insert(body.end(), data, data + len);
                    return true;
                }, nullptr);

            if (!res) {
                fprintf(stderr, "FAILED (HTTP error %d)\n", (int)res.error());
                return false;
            }

            int status = res->status;
            fprintf(stderr, "got %zu bytes, status=%d\n", body.size(), status);

            if (status != 200 && status != 206) {
                fprintf(stderr, "FAILED (HTTP %d)\n", status);
                return false;
            }

            struct gguf_init_params iparams = { /*.no_alloc =*/ 1, /*.ctx =*/ NULL };
            struct gguf_context * gctx = gguf_init_from_buffer(
                body.data(), body.size(), iparams);

            if (gctx) {
                int64_t n_kv = gguf_get_n_kv(gctx);
                int64_t n_tensors = gguf_get_n_tensors(gctx);
                uint32_t ver = gguf_get_version(gctx);
                gguf_free(gctx);

                fprintf(stderr, "Header parsed: v%u, %lld KV, %lld tensors\n",
                        ver, (long long)n_kv, (long long)n_tensors);

                out_data = std::move(body);
                return true;
            }

            fprintf(stderr, "Header incomplete, retrying with larger range...\n");
        }

        fprintf(stderr, "ERROR: could not parse GGUF header within %lld MiB\n",
                (long long)(MAX_CHUNK / (1024 * 1024)));
        return false;
    } catch (const std::exception & e) {
        fprintf(stderr, "HTTP error: %s\n", e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.gguf | https://.../model.gguf> [output.meta.gguf]\n", argv[0]);
        return 1;
    }

    const char * fin_name = argv[1];
    bool is_url = (strncmp(fin_name, "http://", 7) == 0 ||
                   strncmp(fin_name, "https://", 8) == 0);

    // Decide output name
    std::string out_name;
    if (argc >= 3) {
        out_name = argv[2];
    } else {
        out_name = fin_name;
        if (is_url) {
            auto slash = out_name.rfind('/');
            if (slash != std::string::npos) out_name = out_name.substr(slash + 1);
            auto qm = out_name.find('?');
            if (qm != std::string::npos) out_name = out_name.substr(0, qm);
        }
        out_name += ".meta.gguf";
    }

    // ---- case 1: URL download ----

    if (is_url) {
        std::vector<uint8_t> header_data;
        if (!download_header(fin_name, header_data)) return 1;

        struct gguf_init_params iparams = { /*.no_alloc =*/ 1, /*.ctx =*/ NULL };
        struct gguf_context * gctx = gguf_init_from_buffer(
            header_data.data(), header_data.size(), iparams);
        if (!gctx) {
            fprintf(stderr, "ERROR: failed to parse downloaded header\n");
            return 1;
        }

        fprintf(stderr, "Parsed v%u header: %lld KV, %lld tensors\n",
                gguf_get_version(gctx),
                (long long)gguf_get_n_kv(gctx),
                (long long)gguf_get_n_tensors(gctx));

        if (!gguf_write_to_file(gctx, out_name.c_str(), /*only_meta =*/ 1)) {
            fprintf(stderr, "ERROR: failed to write '%s'\n", out_name.c_str());
            gguf_free(gctx);
            return 1;
        }

        FILE * fout = ggml_fopen(out_name.c_str(), "rb");
        if (fout) {
            fseek(fout, 0, SEEK_END);
            long sz = ftell(fout);
            fclose(fout);
            fprintf(stderr, "Wrote metadata: %s (%.1f KiB)\n",
                    out_name.c_str(), sz / 1024.0);
        }

        gguf_free(gctx);
        return 0;
    }

    // ---- case 2: local file ----

    struct gguf_init_params iparams = { /*.no_alloc =*/ 1, /*.ctx =*/ NULL };
    struct gguf_context * gctx = gguf_init_from_file(fin_name, iparams);
    if (!gctx) {
        fprintf(stderr, "ERROR: failed to parse '%s'\n", fin_name);
        return 1;
    }

    fprintf(stderr, "Parsed v%u header: %lld KV, %lld tensors\n",
            gguf_get_version(gctx),
            (long long)gguf_get_n_kv(gctx),
            (long long)gguf_get_n_tensors(gctx));

    if (!gguf_write_to_file(gctx, out_name.c_str(), /*only_meta =*/ 1)) {
        fprintf(stderr, "ERROR: failed to write '%s'\n", out_name.c_str());
        gguf_free(gctx);
        return 1;
    }

    {
        FILE * fout = ggml_fopen(out_name.c_str(), "rb");
        if (fout) {
            fseek(fout, 0, SEEK_END);
            long sz = ftell(fout);
            fclose(fout);
            fprintf(stderr, "Wrote metadata: %s (%.1f KiB)\n",
                    out_name.c_str(), sz / 1024.0);
        }
    }

    gguf_free(gctx);
    return 0;
}

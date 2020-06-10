#include <torch/torch.h>
#include <sgl/sgl.h>
#include <iostream>
#include <optional>

namespace sgl {
namespace dssm {
struct dssm_config {
    int64_t fanin;
    int64_t middle;
    int64_t fanout;
    float dropout;
};

struct parser_config {
    std::string outer_separator = "\n";
    std::string inner_separator = "\t";
    std::string array_separator = "|";
    size_t mini_batch_size = 1024;
    size_t query_group_size = 5;
};

struct training_config {
    float learning_rate;
    size_t epochs;
};

struct application_config {
    sgl::dssm::dssm_config dssm_config;
    sgl::dssm::parser_config parser_config;
    sgl::dssm::training_config training_config;
    std::string input_sample;
    std::string output;
};

template<typename O, typename T>
O& operator<<(O& o, const std::vector<T>& value) {
    for (const auto& x : value) {
        o << x << ' ';
    }
    return o;
}

struct dssm : torch::nn::Module {
public:
    struct mlp : torch::nn::Module  {
        dssm_config config_;
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
        torch::nn::BatchNorm1d bn{nullptr};

    public:
        mlp() = default;
        mlp(const dssm_config& config) : config_(config)  {
            fc1 = register_module("fc1", torch::nn::Linear(config_.fanin, config_.middle));
            fc2 = register_module("fc2", torch::nn::Linear(config_.middle, config_.middle));
            fc3 = register_module("fc3", torch::nn::Linear(config_.middle, config_.fanout));
            bn = register_module("bn", torch::nn::BatchNorm1d(config_.middle));
        }
        mlp(mlp&&) = default;
        mlp(const mlp&) = delete;

        torch::Tensor forward(const torch::Tensor& a) {
            auto x = torch::selu(bn->forward(fc1->forward(a)));
            x = torch::dropout(x, config_.dropout, is_training());
            x = torch::selu(fc2->forward(x));
            return torch::selu(fc3->forward(x));
        }
    }; // class mlp

private:
    dssm_config config_;
    std::shared_ptr<dssm::mlp> mlp_left = nullptr;
    std::shared_ptr<dssm::mlp> mlp_right = nullptr;

public:
    dssm() = default;
    dssm(const dssm_config& config) : config_(config), mlp_left(std::make_shared<dssm::mlp>(config)), mlp_right(std::make_shared<dssm::mlp>(config)) {
        register_module("mlp_left", mlp_left);
        register_module("mlp_right", mlp_right);
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y) {
        return torch::cosine_similarity(mlp_left->forward(x), mlp_right->forward(y));
    }

    dssm_config config() const {
        return config_;
    }
}; // class dssm

torch::Tensor nll(const torch::Tensor& prediction, int64_t group_size) {
    int64_t n = prediction.sizes()[0];
    int64_t rows = n / group_size;
    torch::Tensor exp = torch::exp(prediction).reshape({rows, group_size});
    auto slice = exp.slice(1, 0, 1).reshape(rows);
    auto likelyhood = slice / exp.sum(1);
    likelyhood = likelyhood.reshape(rows);
    return -torch::mean(torch::log(likelyhood));
}

template<typename Optimizer>
requires(sgl::concepts::optimizer(Optimizer))
torch::Tensor fit(Optimizer& optimizer, sgl::dssm::dssm& model, const torch::Tensor& x0, const torch::Tensor& x1, uint64_t group_size) {
    optimizer.zero_grad();
    torch::Tensor prediction = model.forward(x0, x1);
    torch::Tensor loss = sgl::dssm::nll(prediction, group_size);
    loss.backward();
    optimizer.step();
    return loss;
}

inline
bool is_continuation(char x) noexcept {
    return (x & 0xc0) == 0x80;
}

inline
bool is_continuation(char const* first, char const* last) noexcept {
    while (first != last) {
        if (!is_continuation(*first)) {
            return first;
        }
        ++first;
    }
    return first;
}


inline bool is_utf8(uint8_t x) {
    return x < 0x80;
}

inline bool is_utf8(uint8_t b0, uint8_t b1) {
return (b0 & 0xe0u) == 0xc0 && is_continuation(b1) && (b0 & 0x1eu) != 0;
}

inline bool is_utf8(uint8_t b0, uint8_t b1, uint8_t b2) {
    return (b0 & 0xf0u) == 0xe0 && is_continuation(b1) && is_continuation(b2) &&
           !((b0 & 0x0fu) == 0 && (b1 & 0x20u) == 0) && !(b0 == 0xed && b1 & 0x20u);
}

inline bool is_utf8(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (b0 & 0xf8u) == 0xf0 && !(!is_continuation(b1)) && is_continuation(b2) && is_continuation(b3) &&
           !((b0 & 0x07u) == 0 && (b1 & 0x30u) == 0) && !(b0 > 0xf4 || (b0 == 0xf4 && b1 > 0x8f));
}


template<typename It>
inline
It next_utf8(It first, It last) {
    if (first < last && is_utf8(first[0])) {
        return first + 1;
    } else if (first + 1 < last && is_utf8(first[0], first[1])) {
        return first + 2;
    } else if (first + 2 < last && is_utf8(first[0], first[1], first[2])) {
        return first + 3;
    } else if (first + 3 < last && is_utf8(first[0], first[1], first[2], first[3])) {
        return first + 4;
    }
    return first;
}

template<typename It>
inline
std::pair<It, size_t> next_utf8(It first, It last, size_t n) {
    while (first != last && n != 0) {
        --n;
        first = next_utf8(first, last);
    }
    return {first, n};
}



template<typename It, typename F>
It unicode_char_ngrams(It first, It last, size_t n, F function) {
    auto current_last = next_utf8(first, last, n);
    if (current_last.second != 0) {
        return first;
    }
    auto l1 = current_last.first;
    function(first, l1);
    
    while (l1 != last) {
        auto l2 = next_utf8(l1, last);
        if (l2 == l1) { return first; }
        first = next_utf8(first, last);
        function(first, l2);
        l1 = l2;
    }
    return last;
};

template<typename It>
struct unicode_char_ngrams_view {
    It first;
    It last;
    size_t n;

    unicode_char_ngrams_view(It first, It last, size_t n) : first(first), last(last), n(n) {}
    unicode_char_ngrams_view(const unicode_char_ngrams_view&) = default;

    template<typename S>
    friend
    inline
    sgl::v1::unordered_registry<std::string, size_t>& operator<<(sgl::v1::unordered_registry<std::string, S>& registry, const unicode_char_ngrams_view& source) {
        sgl::dssm::unicode_char_ngrams(source.first, source.last, source.n, [&registry](auto first, auto last) {
            registry.push(std::string(first, last));
        });
        return registry;
    }

    template<typename S>
    friend
    inline
    std::unordered_map<std::string, size_t>& operator<<(std::unordered_map<std::string, S>& vocabulary, const unicode_char_ngrams_view& source) {
        sgl::dssm::unicode_char_ngrams(source.first, source.last, source.n, [&vocabulary](auto first, auto last) {
            ++vocabulary[std::string(first, last)];
        });
        return vocabulary;
    }

};


struct ranking {
    sgl::dssm::parser_config parser_config_;
    sgl::dssm::dssm dssm_;
    torch::optim::Adam optimizer_;
    sgl::v1::unordered_registry<std::string, size_t> trigrams_registry;

    std::deque<std::vector<float>> queries;
    std::deque<std::vector<float>> documents;
    float loss_;

    float loss() const {
        return loss_;
    }

    template<typename It>
    bool operator()(It first, It last) {
        It query_first = first;
        It query_last = std::find(query_first, last, parser_config_.inner_separator[0]);
        push_back(query_first, query_last, queries);

        It doc_first = query_last;
        if (doc_first != last) ++doc_first;
        It doc_last = std::find(doc_first, last, parser_config_.inner_separator[0]);
        push_back(doc_first, doc_last, documents);


        int64_t vector_size = trigrams_registry.max_index();
        size_t n = parser_config_.query_group_size * parser_config_.mini_batch_size;
        if (queries.size() < n) {
            return false;
        }

        sgl::v1::array<float> tensor_x(n * trigrams_registry.max_index());
        auto pos_x = tensor_x.begin();

        sgl::v1::array<float> tensor_y(n * trigrams_registry.max_index());
        auto pos_y = tensor_y.begin();

        for (size_t i = 0; i < n; i += parser_config_.query_group_size) {
            for (size_t j = 0; j < parser_config_.query_group_size; ++j) {
                pos_x = std::copy(queries[i].begin(), queries[i].end(), pos_x);
                pos_y = std::copy(documents[i + j].begin(), documents[i + j].end(), pos_y);
            }
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);

        torch::Tensor input_tensor_x = torch::from_blob(tensor_x.data(), int64_t(tensor_x.size()), options);
        input_tensor_x = input_tensor_x.reshape({int64_t(n), vector_size});

        torch::Tensor input_tensor_y = torch::from_blob(tensor_y.data(), int64_t(tensor_y.size()), options);
        input_tensor_y = input_tensor_y.reshape({int64_t(n), vector_size});
    
        // std::cout << input_tensor_x << std::endl << input_tensor_y << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        loss_ = sgl::dssm::fit(optimizer_, dssm_, input_tensor_x, input_tensor_y, parser_config_.query_group_size).item<float>();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Torch took: " << std::chrono::duration<float>(end - start).count() << " sec.; ";
        queries.pop_front();
        documents.pop_front();
        return true;
    }

    template<typename It>
    void push_back(It first, It last, std::deque<std::vector<float>>& queue) {
        std::vector<float> query_vector(trigrams_registry.max_index(), 0.0);
        auto self = this;
        sgl::v1::for_each_split(first, last, parser_config_.array_separator[0], [&self, &query_vector](auto first, auto last) {
            sgl::dssm::unicode_char_ngrams(first, last, 3, [&self, &query_vector](auto first, auto last) {
                auto [term_id, ok] = self->trigrams_registry.get(std::string(first, last));
                if (ok) {
                    query_vector[term_id] = 1;
                }
            });
        });
        queue.emplace_back(std::move(query_vector));
    }

    void save(const std::string& path) {
        torch::save(std::make_shared<sgl::dssm::dssm>(dssm_), path.data());
    }
};

void test() {
    sgl::dssm::dssm_config config{10240, 512, 128, 0.5};
    auto dssm = std::make_shared<sgl::dssm::dssm>(config);
    //torch::Tensor input = torch::ones({4, config.fanin});
    //torch::Tensor similarity = dssm->forward(input);
    float learning_rate = 0.001;
    torch::optim::Adam optimizer(dssm->parameters(), learning_rate);
    int64_t group_size = 1024;
    torch::Tensor data = torch::rand({group_size, 10240});
    auto start = std::chrono::high_resolution_clock::now();
    sgl::dssm::fit(optimizer, *dssm, data, data, group_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<float>(end - start).count() << std::endl;
    std::cout << *dssm << std::endl;

    torch::from_blob(std::vector<int>{1, 2, 3}.data(), 3);
}

void train(sgl::dssm::application_config application_config) {
    sgl::v1::stopwatch_nanoseconds sw;
    auto data = sgl::v1::fmmap<char>(application_config.input_sample.data());

    sgl::v1::unordered_registry<std::string, size_t> trigrams_registry;
    std::unordered_map<std::string, size_t> counts;

    sgl::v1::for_each_split3(
        data.begin(),
        data.end(),
        application_config.parser_config.outer_separator[0],
        application_config.parser_config.inner_separator[0],
        application_config.parser_config.array_separator[0],
        [&counts](auto first, auto last) { counts << sgl::dssm::unicode_char_ngrams_view(first, last, 3); }
    );

    std::vector<std::pair<size_t, std::string>> counts_array;
    counts_array.reserve(counts.size());
    std::transform(counts.begin(), counts.end(), std::back_inserter(counts_array), [](auto& x) { return std::make_pair(x.second, std::move(x.first)); });
    std::sort(counts_array.begin(), counts_array.end(), [](const auto& x, const auto& y) { return y.first < x.first; });
    counts_array.resize(std::min(counts_array.size(), size_t(application_config.dssm_config.fanin)));
    for (const auto& frequency_and_word : counts_array) {
        trigrams_registry.push(frequency_and_word.second);
    }
    application_config.dssm_config.fanin = trigrams_registry.size(); 
    counts.clear();
    std::cerr << "Preprocessing took " << sw.stop() / 1.0e9 << " sec.\n";
    std::cout << "Fanin size: " << application_config.dssm_config.fanin << std::endl;

    sgl::dssm::dssm dssm(application_config.dssm_config);
    torch::optim::Adam optimizer(dssm.parameters(), application_config.training_config.learning_rate);
    sgl::dssm::ranking ranking{application_config.parser_config, std::move(dssm), std::move(optimizer), std::move(trigrams_registry)};

    size_t iteration = 0;
    for (size_t i = 0; i < application_config.training_config.epochs; ++i) {
        sgl::v1::for_each_split(
            data.begin(),
            data.end(),
            application_config.parser_config.outer_separator[0],
            [&](auto first, auto last) {
                sgl::v1::stopwatch_nanoseconds sw;
                if (ranking(first, last)) {
                    auto time_took = sw.stop();
                    std::cerr << "Iteration took: " << time_took / 1.0e9 << "sec.; "
                              << "Epoch " << i << "; "
                              << "Iteration " << ++iteration << "; "
                              << "Loss: " << ranking.loss() << std::endl;
                    if (iteration % 128 == 0) {
                        ranking.save(application_config.output);
                    }
                }
            }
        );
    }
    ranking.save(application_config.output);
}

} // namespace dssm
} // namespace sgl

int main(int argc, const char* argv[]) {
   try {
        sgl::v1::argparser argparser(argc, argv);
        auto [input, input_error] = argparser.get<std::string>("-i,--input");
        if (input.empty() || input_error) {
            std::cerr << "-i,--input is required" << std::endl;
            return 1;
        }
        auto [epochs, epochs_error] = argparser.get<size_t>("-e,--epochs", 5);
        if (epochs_error) {
            std::cerr << "-e,--epochs is an integer" << std::endl;
            return 1;
        }
        auto [output, output_error] = argparser.get<std::string>("-o,--output");
        if (output.empty() || output_error) {
            std::cerr << "-o,--output is required" << std::endl;
            return 1;
        }
        auto [max_fanin, max_fanin_error] = argparser.get<size_t>("--max-fanin", 30000ul);
        if (max_fanin_error) {
            std::cerr << "--max-fanin is an integer" << std::endl;
            return 1;
        }

        sgl::dssm::application_config application_config;
        application_config.dssm_config.fanin = max_fanin;
        application_config.dssm_config.middle = 512;
        application_config.dssm_config.fanout = 128;
        application_config.dssm_config.dropout = 0.5f;
        application_config.training_config.epochs = epochs;
        application_config.training_config.learning_rate = 0.001f;
        application_config.input_sample = input;
        application_config.output = output;

        sgl::dssm::train(application_config);
    } catch(const std::exception& exception) {
        std::cerr << exception.what() << std::endl;
        return 2;
    }
}

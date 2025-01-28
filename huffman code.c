#include <iostream>
#include <queue>
#include <vector>
using namespace std;

struct Minnode {
    unsigned freq;
    char item;
    Minnode *left, *right;
};

Minnode* newNode(char item, unsigned freq) {
    return new Minnode({freq, item, nullptr, nullptr});
}

void printCodes(Minnode* root, vector<int>& code) {
    if (!root) return;

    if (root->item != '$') {
        cout << root->item << " | ";
        for (int i : code) cout << i;
        cout << endl;
    }

    vector<int> left_code = code, right_code = code;
    left_code.push_back(0);
    right_code.push_back(1);

    printCodes(root->left, left_code);
    printCodes(root->right, right_code);
}

struct Compare {
    bool operator()(Minnode* left, Minnode* right) const {
        return left->freq > right->freq;
    }
};

Minnode* bht(char item[], int freq[], int size) {
    priority_queue<Minnode*, vector<Minnode*>, Compare> minHeap;

    for (int i = 0; i < size; ++i)
        minHeap.push(newNode(item[i], freq[i]));

    while (minHeap.size() > 1) {
        Minnode *left = minHeap.top(); minHeap.pop();
        Minnode *right = minHeap.top(); minHeap.pop();

        Minnode *top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;

        minHeap.push(top);
    }
    return minHeap.top();
}

void HuffmanCodes(char item[], int freq[], int size) {
    Minnode* root = bht(item, freq, size);
    vector<int> code;
    printCodes(root, code);
}

int main() {
    int n;
    cout << "Enter the number of characters: ";
    cin >> n;

    char arr[n];
    int freq[n];

    cout << "Enter characters and their frequencies:\n";
    for (int i = 0; i < n; ++i)
        cin >> arr[i] >> freq[i];

    cout << "Char | Huffman code\n----------------------\n";
    HuffmanCodes(arr, freq, n);

    return 0;
}

import time
import re
import os
import yaml
import unicodedata
import string

class KeywordProcessor(object):
    """KeywordProcessor
    Note:
        * Based on Flashtext <https://github.com/vi3k6i5/flashtext>
        * loosely based on `Aho-Corasick algorithm <https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm>`.
    """

    def __init__(self, case_sensitive=False):
        """
        Args:
            case_sensitive (boolean): Keyword search should be case sensitive set or not.
                Defaults to False
        """
        self._keyword = '_keyword_'
        self.non_word_boundaries = set(string.digits + string.ascii_letters + '_')
        self.keyword_trie_dict = dict()
        self.case_sensitive = case_sensitive

    def __setitem__(self, keyword, clean_name, punctuation=None):
        """To add keyword to the dictionary
        pass the keyword and the clean name it maps to.
        Args:
            keyword : string
                keyword that you want to identify
            clean_name : string
                clean term for that keyword that you would want to get back in return or replace
                if not provided, keyword will be used as the clean name also.
            puctuation : list[char]
                list of punctuation characters to add to the keyword before adding.
        """
        if punctuation is None:
            punctuation = ['']
        status = False

        if keyword and clean_name:
            if not self.case_sensitive:
                keyword = keyword.lower()
            current_dict = self.keyword_trie_dict
            for letter in keyword:
                current_dict = current_dict.setdefault(letter, {})
            for punc in punctuation:
                if len(punc) > 0:
                    final_dict = current_dict.setdefault(punc, {})
                else:
                    final_dict = current_dict
                final_dict[self._keyword] = clean_name + punc
            status = True
        return status

    def add_keyword(self, keyword, clean_name, punctuation=None):
        """To add one or more keywords to the dictionary
        pass the keyword and the clean name it maps to.
        Args:
            keyword : string
                keyword that you want to identify
            clean_name : string
                clean term for that keyword that you would want to get back in return or replace
                if not provided, keyword will be used as the clean name also.
            punctuation : list[char]
                list of punctuation characters to add to the keyword before adding.
        Returns:
            status : bool
                The return value. True for success, False otherwise.
        """
        return self.__setitem__(keyword, clean_name, punctuation)

    def replace_keywords(self, sentence):
        """Searches in the string for all keywords present in corpus.
        Keywords present are replaced by the clean name and a new string is returned.
        Args:
            sentence (str): Line of text where we will replace keywords
        Returns:
            new_sentence (str): Line of text with replaced keywords
        """
        if not sentence:
            # if sentence is empty or none just return the same.
            return sentence
        new_sentence = []
        orig_sentence = sentence
        if not self.case_sensitive:
            sentence = sentence.lower()
        current_word = ''
        current_dict = self.keyword_trie_dict
        sequence_end_pos = 0
        idx = 0
        sentence_len = len(sentence)
        while idx < sentence_len:
            char = sentence[idx]
            # when we reach whitespace
            if char not in self.non_word_boundaries:
                current_word += orig_sentence[idx]
                current_white_space = char
                # if end is present in current_dict
                if self._keyword in current_dict or char in current_dict:
                    # update longest sequence found
                    longest_sequence_found = None
                    is_longer_seq_found = False
                    if self._keyword in current_dict:
                        longest_sequence_found = current_dict[self._keyword]
                        sequence_end_pos = idx

                    # re look for longest_sequence from this position
                    if char in current_dict:
                        current_dict_continued = current_dict[char]
                        current_word_continued = current_word
                        idy = idx + 1
                        while idy < sentence_len:
                            inner_char = sentence[idy]
                            if inner_char not in self.non_word_boundaries and self._keyword in current_dict_continued:
                                current_word_continued += orig_sentence[idy]
                                # update longest sequence found
                                current_white_space = inner_char
                                longest_sequence_found = current_dict_continued[self._keyword]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                            if inner_char in current_dict_continued:
                                current_word_continued += orig_sentence[idy]
                                current_dict_continued = current_dict_continued[inner_char]
                            else:
                                break
                            idy += 1
                        else:
                            # end of sentence reached.
                            if self._keyword in current_dict_continued:
                                # update longest sequence found
                                current_white_space = ''
                                longest_sequence_found = current_dict_continued[self._keyword]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                        if is_longer_seq_found:
                            idx = sequence_end_pos
                            current_word = current_word_continued
                    current_dict = self.keyword_trie_dict
                    if longest_sequence_found:
                        new_sentence.append(longest_sequence_found + current_white_space)
                        current_word = ''
                    else:
                        new_sentence.append(current_word)
                        current_word = ''
                else:
                    # we reset current_dict
                    current_dict = self.keyword_trie_dict
                    new_sentence.append(current_word)
                    current_word = ''
            elif char in current_dict:
                # we can continue from this char
                current_word += orig_sentence[idx]
                current_dict = current_dict[char]
            else:
                current_word += orig_sentence[idx]
                # we reset current_dict
                current_dict = self.keyword_trie_dict
                # skip to end of word
                idy = idx + 1
                while idy < sentence_len:
                    char = sentence[idy]
                    current_word += orig_sentence[idy]
                    if char not in self.non_word_boundaries:
                        break
                    idy += 1
                idx = idy
                new_sentence.append(current_word)
                current_word = ''
            # if we are end of sentence and have a sequence discovered
            if idx + 1 >= sentence_len:
                if self._keyword in current_dict:
                    sequence_found = current_dict[self._keyword]
                    new_sentence.append(sequence_found)
                else:
                    new_sentence.append(current_word)
            idx += 1
        return "".join(new_sentence)


def file_to_list(filename, drop_first=True):
    items = []
    with open(filename, "r", encoding="utf-8") as f:
        if drop_first:
            f.readline()

        for line in f.readlines():
            items.append(line.rstrip())
    return items

class PrivacyFilter:

    def __init__(self):
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.keyword_processor_names = KeywordProcessor(case_sensitive=True)
        self.url_re = None
        self.initialised = False
        self.clean_accents = True
        self.nr_keywords = 0
        self.nlp = None
        self.use_nlp = False
        self.use_wordlist = False
        self.use_re = False
        self.numbers_to_zero = False
        ##### CONSTANTS #####
        self._punctuation = ['.', ',', ' ', ':', ';', '?', '!']
        self._capture_words = ["PROPN", "NOUN"]
        self._nlp_blacklist_entities = ["WORK_OF_ART"]

    def to_string(self):
        return 'PrivacyFiter(clean_accents=' + str(self.clean_accents) + ', use_nlp=' + str(self.use_nlp) + \
               ', use_wordlist=' + str(self.use_wordlist) + ')'

    def file_to_list(self, filename, drop_first=True):
        items_count = 0
        items = []

        with open(filename, "r", encoding="utf-8") as f:
            if drop_first:
                f.readline()

            for line in f.readlines():
                items_count += 1
                line = line.rstrip()
                items.append(line)

        self.nr_keywords += items_count
        return items

    def initialize_from_file(self, filename):

        with open(filename) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        clean_accents = data['clean_accents']
        nlp_filter = data['nlp_filter']
        wordlist_filter = data['wordlist_filter']
        regular_expressions = data['regular_expressions']
        numbers_to_zero = data['numbers_to_zero']
        datadir = data['data_directory']

        fields = {
            os.path.join(datadir, data['firstnames']): {"replacement": "<NAAM>",
                                                        "punctuation": None if nlp_filter else self._punctuation},
            os.path.join(datadir, data['lastnames']): {"replacement": "<NAAM>",
                                                       "punctuation": None if nlp_filter else self._punctuation},
            os.path.join(datadir, data['places']): {"replacement": "<PLAATS>", "punctuation": None},
            os.path.join(datadir, data['streets']): {"replacement": "<ADRES>", "punctuation": None},
            os.path.join(datadir, data['diseases']): {"replacement": "<AANDOENING>", "punctuation": None},
            os.path.join(datadir, data['medicines']): {"replacement": "<MEDICIJN>", "punctuation": None},
            os.path.join(datadir, data['nationalities']): {"replacement": "<NATIONALITEIT>", "punctuation": None},
            os.path.join(datadir, data['countries']): {"replacement": "<LAND>", "punctuation": None},
        }

        self.initialize(clean_accents=clean_accents,
                        nlp_filter=nlp_filter,
                        wordlist_filter=wordlist_filter,
                        regular_expressions=regular_expressions,
                        numbers_to_zero=numbers_to_zero,
                        fields=fields)

    def initialize(self, clean_accents=True, nlp_filter=True, wordlist_filter=False,
                   regular_expressions=True, numbers_to_zero=False, fields=None):

        # Add words with an append character to prevent replacing partial words by tags.
        # E.g. there is a street named AA and a verb AABB, with this additional character
        # would lead to <ADRES>BB which is incorrect. Another way to solve this might be the
        # implementation of a token based algorithm.
        if not fields:
            fields = {
                os.path.join('datasets', 'firstnames.csv'): {"replacement": "<NAAM>",
                                                             "punctuation": None if nlp_filter else self._punctuation},
                os.path.join('datasets', 'lastnames.csv'): {"replacement": "<NAAM>",
                                                            "punctuation": None if nlp_filter else self._punctuation},
                os.path.join('datasets', 'places.csv'): {"replacement": "<PLAATS>", "punctuation": None},
                os.path.join('datasets', 'streets_Nederland.csv'): {"replacement": "<ADRES>", "punctuation": None},
                os.path.join('datasets', 'diseases.csv'): {"replacement": "<AANDOENING>", "punctuation": None},
                os.path.join('datasets', 'medicines.csv'): {"replacement": "<MEDICIJN>", "punctuation": None},
                os.path.join('datasets', 'nationalities.csv'): {"replacement": "<NATIONALITEIT>", "punctuation": None},
                os.path.join('datasets', 'countries.csv'): {"replacement": "<LAND>", "punctuation": None},
            }

        for field in fields:
            # If there is a punctuation list, use it.
            if fields[field]["punctuation"] is not None:
                for name in self.file_to_list(field):
                    for c in self._punctuation:
                        self.keyword_processor.add_keyword(
                            "{n}{c}".format(n=name, c=c),
                            "{n}{c}".format(n=fields[field]["replacement"], c=c)
                        )
            else:
                for name in self.file_to_list(field):
                    self.keyword_processor.add_keyword(name, fields[field]["replacement"])

        if not nlp_filter:
            for name in self.file_to_list(os.path.join('datasets', 'firstnames.csv')):
                self.keyword_processor_names.add_keyword(name, "<NAAM>")

            for name in self.file_to_list(os.path.join('datasets', 'lastnames.csv')):
                self.keyword_processor_names.add_keyword(name, "<NAAM>")

        # Make the URL regular expression
        # https://stackoverflow.com/questions/827557/how-do-you-validate-a-url-with-a-regular-expression-in-python
        ul = '\u00a1-\uffff'  # Unicode letters range (must not be a raw string).

        # IP patterns
        ipv4_re = r'(?:0|25[0-5]|2[0-4]\d|1\d?\d?|[1-9]\d?)(?:\.(?:0|25[0-5]|2[0-4]\d|1\d?\d?|[1-9]\d?)){3}'
        ipv6_re = r'\[?((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,'\
                  r'4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{'\
                  r'1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2['\
                  r'0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,'\
                  r'3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|['\
                  r'1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,'\
                  r'2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|((['\
                  r'0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2['\
                  r'0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:['\
                  r'0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2['\
                  r'0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,'\
                  r'5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?\]?'

        # Host patterns
        hostname_re = r'[a-z' + ul + r'0-9](?:[a-z' + ul + r'0-9-]{0,61}[a-z' + ul + r'0-9])?'
        # Max length for domain name labels is 63 characters per RFC 1034 sec. 3.1
        domain_re = r'(?:\.(?!-)[a-z' + ul + r'0-9-]{1,63}(?<!-))*'
        tld_re = (
                r'\.'                                # dot
                r'(?!-)'                             # can't start with a dash
                r'(?:[a-z' + ul + '-]{2,63}'         # domain label
                r'|xn--[a-z0-9]{1,59})'              # or punycode label
                r'(?<!-)'                            # can't end with a dash
                r'\.?'                               # may have a trailing dot
        )
        host_re = '(' + hostname_re + domain_re + tld_re + '|localhost)'

        self.url_re = re.compile(
            r'([a-z0-9.+-]*:?//)?'                                       # scheme is validated separately
            r'(?:[^\s:@/]+(?::[^\s:@/]*)?@)?'                           # user:pass authentication
            r'(?:' + ipv4_re + '|' + ipv6_re + '|' + host_re + ')'
            r'(?::\d{2,5})?'                                            # port
            r'(?:[/?#][^\s]*)?',                                        # resource path
            re.IGNORECASE
        )
        self.use_wordlist = wordlist_filter
        self.clean_accents = clean_accents
        self.use_re = regular_expressions
        self.numbers_to_zero = numbers_to_zero

        self.initialised = True

    @staticmethod
    def remove_numbers(text, numbers_to_zero):
        if numbers_to_zero:
            return re.sub('\d', '0', text).strip()
        else:
            return re.sub(r'\w*\d\w*', '<GETAL>', text).strip()

    @staticmethod
    def remove_times(text):
        return re.sub('(\d{1,2})[.:](\d{1,2})?([ ]?(am|pm|AM|PM))?', '<TIJD>', text)

    @staticmethod
    def remove_dates(text):
        text = re.sub("\d{2}[- /.]\d{2}[- /.]\d{,4}", "<DATUM>", text)

        text = re.sub(
            "(\d{1,2}[^\w]{,2}(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)"
            "([- /.]{,2}(\d{4}|\d{2}))?)",
            "<DATUM>", text)

        text = re.sub(
            "(\d{1,2}[^\w]{,2}(jan|feb|mrt|apr|mei|jun|jul|aug|sep|okt|nov|dec))[- /.](\d{4}|\d{2})?",
            "<DATUM>", text)
        return text

    @staticmethod
    def remove_email(text):
        return re.sub("(([a-zA-Z0-9_+]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))"
                      "(?![^<]*>)",
                      "<EMAIL>",
                      text)

    def remove_url(self, text):
        text = re.sub(self.url_re, "<URL>", text)
        return text

    @staticmethod
    def remove_postal_codes(text):
        return re.sub(r"\b([0-9]{4}\s?[a-zA-Z]{2})\b", "<POSTCODE>", text)

    @staticmethod
    def remove_accents(text):
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
        return str(text.decode("utf-8"))

    def filter_keyword_processors(self, text):
        text = self.keyword_processor.replace_keywords(text)
        text = self.keyword_processor_names.replace_keywords(text)
        return text

    def filter_regular_expressions(self, text):
        text = self.remove_url(text)
        text = self.remove_dates(text)
        text = self.remove_times(text)
        text = self.remove_email(text)
        text = self.remove_postal_codes(text)
        text = self.remove_numbers(text, self.numbers_to_zero)
        return text

    def filter_nlp(self, text):
        if not self.nlp:
            self.initialize(clean_accents=self.clean_accents, nlp_filter=True)

        doc = self.nlp(text)  # Run text through NLP

        # Word, tags, word type, entity type
        tagged_words = [(str(word), word.tag_, word.pos_, word.ent_type_) for word in doc]
        tagged_words_new = []

        index = 0
        length = len(tagged_words)
        capture_string = ""
        captured_entity = ""

        for tagged_word in tagged_words:
            word, tags, word_type, entity_type = tagged_word
            is_capture_word = word_type in self._capture_words

            # If it is a capture word, add it to the string to be tested
            if is_capture_word:
                capture_string += "{} ".format(word)
                if entity_type != "" and entity_type not in self._nlp_blacklist_entities:
                    captured_entity = entity_type

            # Check if next word is also forbidden
            if is_capture_word and index + 1 < length:
                next_word = tagged_words[index + 1]
                if next_word[2] in self._capture_words:
                    index += 1
                    continue

            # Filter the collected words if they are captured
            if is_capture_word:
                if captured_entity == "" or captured_entity in self._nlp_blacklist_entities:
                    replaced = self.keyword_processor.replace_keywords(capture_string).strip()
                else:
                    replaced = "<{}>".format(captured_entity)

            elif word_type == "NUM":
                if self.numbers_to_zero:
                    replaced = "0"
                else:
                    replaced = "<GETAL>"
            else:
                replaced = word

            # Replace the word, even if it wasn't replaced
            tagged_words_new.append((replaced, tags, word_type, captured_entity))

            index += 1
            capture_string = ""
            captured_entity = ""

        # Rebuild the string from the filtered output
        new_string = ""
        for tagged_word in tagged_words_new:
            word, tags, word_type, entity_type = tagged_word
            new_string += (" " if word_type != "PUNCT" else "") + word  # Prepend spaces, except for punctuation.

        new_string = new_string.strip()
        return new_string

    @staticmethod
    def cleanup_text(result):
        result = re.sub("<[A-Z _]+>", "<FILTERED>", result)
        result = re.sub(" ([ ,.:;?!])", "\\1", result)
        result = re.sub(" +", " ", result)                          # remove multiple spaces
        result = re.sub("\n +", "\n", result)                       # remove space after newline
        result = re.sub("( <FILTERED>)+", " <FILTERED>", result)    # remove multiple consecutive <FILTERED> tags
        return result.strip()

    def filter(self, text):
        if not self.initialised:
            self.initialize()
        text = str(text)
        if self.clean_accents:
            text = self.remove_accents(text)

        if self.use_nlp:
            text = self.filter_nlp(text)
        if self.use_re:
            text = self.filter_regular_expressions(text)

        if self.use_wordlist:
            text = self.filter_static(text)

        return self.cleanup_text(text)

    def filter_static(self, text):
        text = " " + text + " "
        text = self.filter_regular_expressions(text)
        text = self.filter_keyword_processors(text)
        return text


def insert_newlines(string, every=64, window=10):
    """
    Insert a new line every n characters. If possible, break
    the sentence at a space close to the cutoff point.
    Parameters
    ----------
    string Text to adapt
    every Maximum length of each line
    window The window to look for a space

    Returns
    -------
    Adapted string
    """
    result = ""
    from_string = string
    while len(from_string) > 0:
        cut_off = every
        if len(from_string) > every:
            while (from_string[cut_off - 1] != ' ') and (cut_off > (every - window)):
                cut_off -= 1
        else:
            cut_off = len(from_string)
        part = from_string[:cut_off]
        result += part + '\n'
        from_string = from_string[cut_off:]
    return result[:-1]


def main():
    zin = "De mogelijkheden zijn sinds 2014 groot geworden, zeker vergeleken met 2012, hè Kees? Het systeem maakt " \
          "verschillende bewerkingen mogelijk die hiervoor niet mogelijk waren. De datum is 24-01-2011 (of 24 jan 21 " \
          "of 24 januari 2011). Ik ben te bereiken op naam@hostingpartner.nl en woon in Arnhem. Mijn adres is " \
          "Maasstraat 231, 1234AB. Mijn naam is Thomas Janssen en ik heb zweetvoeten. Oh ja, ik gebruik hier " \
          "heparine ( https://host.com/dfgr/dfdew ) voor. Simòne. Ik heet Lexan."

    print(insert_newlines(zin, 120))

    start = time.time()
    pfilter = PrivacyFilter()
    pfilter.initialize_from_file('filter.yaml')
    print('\nInitialisation time       : %4.0f msec' % ((time.time() - start) * 1000))
    print('Number of forbidden words : ' + str(pfilter.nr_keywords))

    start = time.time()
    nr_sentences = 100
    for i in range(0, nr_sentences):
        zin2 = pfilter.filter(zin)

    print('Time per sentence         : %4.2f msec' % ((time.time() - start) * 1000 / nr_sentences))
    print()
    print(insert_newlines(zin2, 120))


if __name__ == "__main__":
    main()

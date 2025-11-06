package collator

import (
	"encoding/json"
	"log"
	"os"
	"strings"
	"sync"
	"time"
)

type Record struct {
	Attendance bool      `json:"attendance"`
	Detected   bool      `json:"detected"`
	FirstSeen  time.Time `json:"firstSeen"`
	LastSeen   time.Time `json:"lastSeen"`
	ReferenceID string   `json:"referenceid"`
}

type Store struct {
	mu     sync.Mutex
	Items  map[string]Record `json:"personnel"`
}

type Person struct {
	Name   string   `json:"name"`
	Images []string `json:"images"`
}

type JsonStruct struct {
	Img_fp  string   `json:"img_folder_path"`
	Details []Person `json:"details"`
}

type StoreCount struct {
	Total    int `json:"total"`
	Detected int `json:"detected"`
	Attended int `json:"attended"`
}

func NewStore() *Store {
	return &Store{Items: make(map[string]Record)}
}

func (store *Store) Check(name string) {
	store.mu.Lock()
	defer store.mu.Unlock()

	currTime := time.Now()

	record, exists := store.Items[name]
	if exists {
		record.Detected = true
		record.LastSeen = currTime

		if record.FirstSeen.IsZero() {
			record.FirstSeen = currTime
			record.Attendance = true
			log.Printf("%s present!", name)
		}

		store.Items[name] = record
	}
}

func (store *Store) Mark(name string) {
	store.mu.Lock()
	defer store.mu.Unlock()

	record, exists := store.Items[name]
	if exists {
		record.Attendance = !record.Attendance
		store.Items[name] = record
	}
}

func (store *Store) Count() StoreCount {
	store.mu.Lock()
	defer store.mu.Unlock()

	detected := 0
	attended := 0

	for _, person := range store.Items {
		if person.Detected {
			detected += 1
		}
		if person.Attendance {
			attended += 1
		}
	}

	count := StoreCount{
		Total:    len(store.Items),
		Detected: detected,
		Attended: attended,
	}

	return count
}

func (store *Store) Add(name string) {
	store.mu.Lock()
	defer store.mu.Unlock()

	var initRecord Record
	initRecord.Detected = false
	initRecord.Attendance = false

	store.Items[name] = initRecord
}

func (store *Store) Clear() {
	store.mu.Lock()
	defer store.mu.Unlock()

	store.Items = make(map[string]Record)
}

func (store *Store) JsonOut() ([]byte, error) {
	store.mu.Lock()
	defer store.mu.Unlock()

	jsonData, err := json.MarshalIndent(store.Items, "", "	")
	if err != nil {
		return nil, err
	}

	return jsonData, nil
}

func (store *Store) JsonSave(filename string) {
	jsonData, err := store.JsonOut()
	if err != nil {
		log.Printf("Error marshaling to JSON: %v", err)
	}

	err = os.WriteFile(filename, jsonData, 0644)
	if err != nil {
		log.Printf("Error writing to file: %v", err)
	}
}

func (store *Store) LoadJSON(bytes []byte) error {
	var jsonData JsonStruct

	// Unmarshal the JSON bytes into the JsonStruct
	if err := json.Unmarshal(bytes, &jsonData); err != nil {
		return err
	}

	// Clear the store
	store.Clear()

	// Loop through the details (person) from the uploaded JSON
	for _, person := range jsonData.Details {
		// Add the person to the store with an initial Record
		store.Add(person.Name)

		// If there are images, set the first image as the ReferenceID
		if len(person.Images) > 0 {
			record := store.Items[person.Name] // Get the record by name
			// Remove ".png" extension from the ReferenceID
			record.ReferenceID = strings.TrimSuffix(person.Images[0], ".png")
			store.Items[person.Name] = record    // Update the record in the store
		}
	}

	return nil
}

func (store *Store) LoadPrevOutput(filename string) {
	file, err := os.Open(filename)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("File %s does not exist.", filename)
		} else {
			log.Printf("Error opening file: %v", err)
		}
		return
	}

	defer file.Close()

	decoder := json.NewDecoder(file)

	store.mu.Lock()
	defer store.mu.Unlock()

	if err := decoder.Decode(&store.Items); err != nil {
		log.Printf("Error decoding JSON: %v", err)
		return
	}

	log.Printf("Loaded data from previous session")
}

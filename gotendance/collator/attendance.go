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
	Name        string    `json:"name"`
	Attendance  bool      `json:"attendance"`
	Detected    bool      `json:"detected"`
	FirstSeen   time.Time `json:"firstSeen"`
	LastSeen    time.Time `json:"lastSeen"`
	ReferenceID string    `json:"referenceid"`
	Tags        []string  `json:"tags"`
}

type Store struct {
	mu     sync.Mutex
	Items  map[string]Record `json:"personnel"`
}

type Person struct {
	Name   string   `json:"name"`
	Images []string `json:"images"`
	Tags   []string `json:"tags"`
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
		record.Attendance = true
		
		if record.FirstSeen.IsZero() { // never seen before!0
			record.FirstSeen = currTime
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

func (store *Store) ResetAllAttendance() {
	store.mu.Lock()
	defer store.mu.Unlock()

	for name, record := range store.Items {
		record.Attendance = false
		record.Detected = false
		record.FirstSeen = time.Time{}
		record.LastSeen = time.Time{}
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
	initRecord.Tags = []string{} // Initialize as empty slice instead of nil

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

	store.mu.Lock()
	defer store.mu.Unlock()

	// Store old record data
	oldRecords := make(map[string]Record)
	for name, record := range store.Items {
		oldRecords[name] = record
	}

	// Clear and reload personnel from the new JSON
	store.Items = make(map[string]Record)

	// Loop through the details (person) from the uploaded JSON
	for _, person := range jsonData.Details {
		record := Record{
			Name:       person.Name,
			Tags:       person.Tags,
		}

		// Preserve attendance, detected, and timestamps from previous session if person still exists
		if oldRecord, exists := oldRecords[person.Name]; exists {
			record.Attendance = oldRecord.Attendance
			record.Detected = oldRecord.Detected
			record.FirstSeen = oldRecord.FirstSeen
			record.LastSeen = oldRecord.LastSeen
		} else {
			record.Attendance = false
			record.Detected = false
			record.FirstSeen = time.Time{}
			record.LastSeen = time.Time{}
		}

		// If there are images, set the first image as the ReferenceID
		if len(person.Images) > 0 {
			// Remove ".png" extension from the ReferenceID
			record.ReferenceID = strings.TrimSuffix(person.Images[0], ".png")
		}

		store.Items[person.Name] = record
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

	// Convert nil tags to empty slices
	for name, record := range store.Items {
		if record.Tags == nil {
			record.Tags = []string{}
			store.Items[name] = record
		}
	}

	log.Printf("Loaded data from previous session")
}
